"""Persistent CPU-only spawned simulator workers and exact rollout cache."""
from __future__ import annotations
import concurrent.futures as cf
import multiprocessing as mp
import os
from pathlib import Path
import time
import json
from collections import OrderedDict
import numpy as np
from .contexts import load_contexts, atomic_json, hash_value, source_hash
from .renderer import RobotAdapter, pairwise_motion, adapter_parameters

_WORKER_ADAPTER = None


def _worker_init(context_dir):
    global _WORKER_ADAPTER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError: pass
    _WORKER_ADAPTER = RobotAdapter(load_contexts(context_dir))


def _execute_chunk(chunk):
    return [(key,_WORKER_ADAPTER.rollout(q,context_id)) for key,q,context_id in chunk]


def _noop(): return os.getpid()


class RolloutPool:
    def __init__(self, context_dir, workers=4, chunk_size=64, cache_dir=None):
        self.context_dir = str(Path(context_dir).resolve())
        self.contexts={int(c["context_id"]):c for c in load_contexts(self.context_dir)}
        self.H = next(iter(self.contexts.values()))["H"]
        self.workers=int(workers); self.chunk_size=int(chunk_size)
        self.code_hash=source_hash(); self.params=adapter_parameters(self.H)
        self.prefix={i:hash_value({"context":c["context_hash"],"adapter":self.params,"code":self.code_hash,
                                 "nominal_actions":c["nominal_actions"],"versions":c["versions"]}) for i,c in self.contexts.items()}
        for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
            os.environ[name]="1"
        self.executor=cf.ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context("spawn"),
                                            initializer=_worker_init,initargs=(self.context_dir,))
        self.cache={}; self.scanned=set(); self._loaded=OrderedDict()
        self.total_logical=0; self.total_unique=0; self.total_cache_hits=0
        if cache_dir is not None: self.scan_cache(cache_dir)

    def key(self,q,context_id): return hash_value({"prefix":self.prefix[int(context_id)],"q":np.asarray(q)})

    def scan_cache(self,directory):
        directory=Path(directory).resolve()
        if str(directory) in self.scanned:return
        self.scanned.add(str(directory))
        if not directory.exists():return
        for path in sorted(directory.glob("*.json")):
            if not path.with_suffix(".npz").exists():continue
            try:
                report=json.loads(path.read_text())
                if report.get("adapter_code_hash")!=self.code_hash:continue
                for row,diag in enumerate(report.get("per_candidate",[])):
                    if diag.get("status") in ("executed","cache_hit") and diag.get("simulation_valid"):
                        self.cache.setdefault(diag["cache_key"],(path.with_suffix(".npz"),row,diag))
            except (ValueError,KeyError,OSError):continue

    def _cached_result(self,record):
        path,row,diag=record; name=str(path)
        if name not in self._loaded:
            with np.load(path,allow_pickle=False) as data:
                self._loaded[name]={k:data[k] for k in ("q","q_hat","positions","submitted_actions","gripper_distance","mocap_positions")}
            if len(self._loaded)>3:self._loaded.popitem(last=False)
        self._loaded.move_to_end(name)
        result={key:value[row] for key,value in self._loaded[name].items()}
        result["diagnostics"]=dict(diag)
        return result

    def execute_queries(self,requests,output_prefix):
        """Execute exact q[M,4] requests, persist one shard, return its summary.

        Required keys: request_id,context_id,q. Any other JSONable fields become
        query metadata. Duplicate candidate slots remain in the output arrays.
        A thread may own this pool while a producer uses the GPU concurrently.
        """
        started=time.perf_counter(); requests=list(requests)
        prefix=Path(output_prefix)
        if prefix.suffix in (".json",".npz"):prefix=prefix.with_suffix("")
        prefix.parent.mkdir(parents=True,exist_ok=True); self.scan_cache(prefix.parent)
        jobs={}; slots=[]; logical_hit=[]; cached={}
        for ri,request in enumerate(requests):
            q=np.asarray(request["q"]); cid=int(request["context_id"])
            if q.ndim!=2 or q.shape[1]!=4:raise ValueError("Request q must have shape [M,4]")
            for slot,candidate in enumerate(q):
                key=self.key(candidate,cid)
                is_hit=key in self.cache or key in jobs
                slots.append((ri,slot,key)); logical_hit.append(is_hit)
                if key in self.cache:cached[key]=self.cache[key]
                elif key not in jobs:jobs[key]=(key,candidate.copy(),cid)
        values=list(jobs.values())
        # Keep enough chunks for every worker even on small query shards.
        chunk=min(self.chunk_size,max(1,(len(values)+self.workers-1)//self.workers))
        futures=[self.executor.submit(_execute_chunk,values[i:i+chunk]) for i in range(0,len(values),chunk)]
        executed={}
        for future in cf.as_completed(futures):
            for key,result in future.result():executed[key]=result
        n=len(slots); arrays={"positions":np.empty((n,self.H+1,3)),"submitted_actions":np.empty((n,self.H,4),dtype=np.float32),
                             "q":np.empty((n,4)),"q_hat":np.empty((n,4)),
                             "gripper_distance":np.empty((n,self.H+1)),"mocap_positions":np.empty((n,self.H+1,3)),
                             "steps":np.empty(n,dtype=np.int16),"request_index":np.empty(n,dtype=np.int32),
                             "candidate_slot":np.empty(n,dtype=np.int16),"cache_keys":np.empty(n,dtype="U64")}
        diagnostics=[]
        for row,((ri,slot,key),is_hit) in enumerate(zip(slots,logical_hit)):
            result=executed[key] if key in executed else self._cached_result(cached[key])
            for name in ("positions","submitted_actions","q","q_hat","gripper_distance","mocap_positions"):arrays[name][row]=result[name]
            diag=dict(result["diagnostics"])
            if is_hit and diag["simulation_valid"]:diag["status"]="cache_hit"
            diag.update({"request_id":str(requests[ri]["request_id"]),"candidate_slot":slot,"request_index":ri})
            arrays["steps"][row]=diag["actual_steps"]; arrays["request_index"][row]=ri
            arrays["candidate_slot"][row]=slot;arrays["cache_keys"][row]=key;diagnostics.append(diag)
        query_diags=[]
        for ri,request in enumerate(requests):
            mask=arrays["request_index"]==ri
            extra={key:val for key,val in request.items() if key!="q"}
            query_diags.append({**extra,**pairwise_motion(arrays["q"][mask],arrays["positions"][mask],arrays["steps"][mask])})
        summary={"logical_requests":n,"unique_executions":len(executed),
                 "cache_hits":sum(d["status"]=="cache_hit" for d in diagnostics),
                 "failed":sum(d["status"]=="failed" for d in diagnostics),
                 "simulation_valid":sum(d["simulation_valid"] for d in diagnostics),
                 "environment_action_steps":sum(r["diagnostics"]["actual_steps"] for r in executed.values()),
                 "elapsed_seconds":time.perf_counter()-started,"workers":self.workers,"chunk_size":chunk}
        summary["unique_clips_per_second"]=len(executed)/max(summary["elapsed_seconds"],1e-12)
        report={"adapter_code_hash":self.code_hash,"context_hashes":{str(k):v["context_hash"] for k,v in self.contexts.items()},
                "summary":summary,"per_candidate":diagnostics,"per_query":query_diags,
                "storage":"positions contain initial+50 post-action hand positions; NaN tail means unexecuted steps; float32 submitted actions; no full observation storage"}
        tmp=prefix.with_suffix(f".npz.{os.getpid()}.tmp")
        with tmp.open("wb") as f:np.savez(f,**arrays)
        os.replace(tmp,prefix.with_suffix(".npz"))
        atomic_json(prefix.with_suffix(".json"),report)
        for row,diag in enumerate(diagnostics):
            if diag["simulation_valid"]:self.cache.setdefault(diag["cache_key"],(prefix.with_suffix(".npz"),row,diag))
        self.total_logical+=n;self.total_unique+=len(executed);self.total_cache_hits+=summary["cache_hits"]
        return {**summary,"npz_path":str(prefix.with_suffix(".npz")),"json_path":str(prefix.with_suffix(".json"))}

    def close(self):self.executor.shutdown(wait=True,cancel_futures=False)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def benchmark(context_dir,output_dir,candidates=(4,8,12,16,20,24),clips_per_worker=256):
    """Actual standard env.step clips, independent engineering RNG, no pixels."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count()
    profile=[]; rng=np.random.default_rng(917243)
    for workers in candidates:
        if workers>affinity:continue
        started=time.perf_counter()
        with RolloutPool(context_dir,workers,chunk_size=32) as pool:
            warm=[pool.executor.submit(_execute_chunk,[(f"warm_{i}",rng.uniform(-1,1,4),i%5)]) for i in range(workers)]
            for f in warm:f.result()
            initialization=time.perf_counter()-started
            n=max(256,workers*clips_per_worker)
            queries=[{"request_id":f"profile_w{workers}_q{i}","context_id":i%5,"q":rng.uniform(-1,1,(16,4))} for i in range((n+15)//16)]
            call_start=time.perf_counter()
            result=pool.execute_queries(queries,output_dir/f"workers_{workers}")
            full_seconds=time.perf_counter()-call_start
            result["compute_and_merge_seconds"]=result["elapsed_seconds"]
            result["elapsed_seconds"]=full_seconds
            result["unique_clips_per_second"]=result["unique_executions"]/full_seconds
            result["throughput_includes"]= "env.step, IPC, exact cache, aggregation, NPZ and JSON writes"
            profile.append({"workers":workers,"initialization_seconds":initialization,**result})
            print(json.dumps({"robot_profile":profile[-1]}),flush=True)
    fastest=max(profile,key=lambda item:item["unique_clips_per_second"])
    result={"profiles":profile,"recommended_workers":fastest["workers"],"measured_clips_per_second":fastest["unique_clips_per_second"],"recommended_clips_per_second":fastest["unique_clips_per_second"],
            "selected_candidate_logical_requests":161280,"uncached_selected_estimated_seconds":161280/fastest["unique_clips_per_second"],
            "eta_limitation":"isolated short benchmark, initialization excluded from steady throughput; concurrent inference contention measured separately"}
    atomic_json(output_dir/"robot_profile.json",result);return result


def benchmark_pool(context_dir, output_dir, counts=(4,8,12,16,20,24)):
    return benchmark(context_dir, output_dir, candidates=counts)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--contexts",required=True);p.add_argument("--output",required=True)
    p.add_argument("--workers",nargs="*",type=int,default=[4,8,12,16,20,24]);a=p.parse_args()
    print(json.dumps(benchmark(a.contexts,a.output,a.workers),indent=2))
