"""Mandatory physical audit tests against the actual pinned simulator."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from metaworld_fast.contexts import prepare_contexts, load_contexts
from metaworld_fast.renderer import RobotAdapter, motion_basis, pairwise_motion
from metaworld_fast.rollout_pool import RolloutPool


class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context_dir=Path(__file__).resolve().parents[1]/"engineering"/"contexts"
        cls.contexts=prepare_contexts(cls.context_dir)
        cls.adapter=RobotAdapter(cls.contexts)

    @classmethod
    def tearDownClass(cls):cls.adapter.close()

    def test_action_space_basis_and_float32_reconstruction(self):
        self.assertEqual(self.adapter.env.action_space.shape,(4,))
        np.testing.assert_array_equal(self.adapter.env.action_space.low,-np.ones(4))
        np.testing.assert_array_equal(self.adapter.env.action_space.high,np.ones(4))
        self.assertEqual(np.linalg.matrix_rank(motion_basis().reshape(-1,4)),4)
        for q in (np.zeros(4),np.ones(4)*33,np.array([-33,33,-33,33]),np.array([.13,-.59,.79,-.1])):
            command,q_hat=self.adapter.commands(q,0)
            self.assertEqual(command.dtype,np.float32)
            self.assertLessEqual(np.abs(command[:,:3]).max(),.950001)
            self.assertLess(np.linalg.norm(q-q_hat),1e-4)
        with self.assertRaises(ValueError):self.adapter.commands(np.ones(4)*33.1,0)

    def test_restore_candidate_order_and_context_switch(self):
        q=np.array([.2,-.3,.4,-.5])
        a=self.adapter.rollout(q,0)
        self.adapter.rollout(-q,1)
        self.adapter.rollout(q*10,0)
        b=self.adapter.rollout(q,0)
        np.testing.assert_array_equal(a["positions"],b["positions"])
        np.testing.assert_array_equal(a["submitted_actions"],b["submitted_actions"])
        self.assertTrue(a["diagnostics"]["simulation_valid"])
        self.assertFalse(a["diagnostics"]["renderer_created"])

    def test_restore_matches_original_nominal_execution(self):
        from metaworld_fast.contexts import make_env, restore_state
        env=make_env(self.contexts[0])
        try:
            for context in self.contexts:
                restore_state(env,context["state"])
                positions=[env.get_endeff_pos().copy()]
                for action in context["nominal_actions"]:
                    _,_,terminated,truncated,_=env.step(action)
                    self.assertFalse(terminated or truncated)
                    positions.append(env.get_endeff_pos().copy())
                np.testing.assert_array_equal(positions,context["nominal_positions"])
        finally:env.close()

    def test_never_step_after_truncation(self):
        contexts=copy.deepcopy(self.contexts)
        contexts[0]["state"]["attributes"]["curr_path_length"]=499
        adapter=RobotAdapter(contexts)
        try:result=adapter.rollout(np.ones(4),0)
        finally:adapter.close()
        self.assertEqual(result["diagnostics"]["actual_steps"],1)
        self.assertTrue(result["diagnostics"]["truncated"])
        self.assertFalse(result["diagnostics"]["simulation_valid"])
        self.assertFalse(result["diagnostics"]["command_reconstruction_ok"])
        self.assertTrue(np.isnan(result["submitted_actions"][1:]).all())

    def test_four_coordinate_motion_sensitivity(self):
        jac=[]
        for axis in range(4):
            q=np.eye(4)[axis]*.1
            plus=self.adapter.rollout(q,0);minus=self.adapter.rollout(-q,0)
            jac.append(((plus["positions"][1:]-minus["positions"][1:])/.2).ravel())
        singular=np.linalg.svd(np.stack(jac,axis=1),compute_uv=False)
        self.assertGreater(singular.min(),1e-7)

    def test_exact_cache_dtype_and_nearby_queries(self):
        q=np.zeros(4)
        self.assertNotEqual(self.adapter.key(q,0),self.adapter.key(q.astype(np.float32),0))
        nearby=q.copy();nearby[0]=1e-15
        self.assertNotEqual(self.adapter.key(q,0),self.adapter.key(nearby,0))
        self.assertNotEqual(self.adapter.key(q,0),self.adapter.key(q,1))

    def test_duplicate_slots_and_spawn_worker_count_invariance(self):
        q=np.array([[.1,.2,.3,.4],[.1,.2,.3,.4],[-.1,-.2,-.3,-.4]])
        requests=[dict(request_id="unit",context_id=0,q=q,kind="selected")]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with RolloutPool(self.context_dir,workers=2) as pool:
                first=pool.execute_queries(requests,root/"two"/"first")
                second=pool.execute_queries(requests,root/"two"/"second")
                self.assertEqual(first["logical_requests"],3)
                self.assertEqual(first["unique_executions"],2)
                self.assertEqual(first["cache_hits"],1)
                self.assertEqual(second["unique_executions"],0)
                self.assertEqual(second["cache_hits"],3)
            with RolloutPool(self.context_dir,workers=3) as pool:
                third=pool.execute_queries(requests,root/"three"/"third")
            with np.load(first["npz_path"]) as a,np.load(third["npz_path"]) as b:
                np.testing.assert_array_equal(a["positions"],b["positions"])
                pairs=pairwise_motion(a["q"],a["positions"],a["steps"])
                self.assertEqual(pairs["coordinate_pair_distance"][0],0)
                self.assertEqual(pairs["motion_pairwise_rms"][0],0)
            with RolloutPool(self.context_dir,workers=2) as pool:
                resumed=pool.execute_queries(requests,root/"two"/"resumed")
                self.assertEqual(resumed["unique_executions"],0)


if __name__=="__main__":unittest.main()
