import unittest
import tempfile
from pathlib import Path
import numpy as np

from src.mnl_inference import validate, likelihood_parts, fit, covariance
from research_inference.run import compare_reference, acquire_input, run, ROOT


class InferenceChecks(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros((8, 2, 1))
        self.x[:, 0, 0] = 1
        self.av = np.ones((8, 2), dtype=int)
        self.choice = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        self.group = np.repeat(np.arange(4), 2)
        self.data = validate(self.x, self.av, self.choice, self.group)
        self.beta = np.array([np.log(3)])

    def test_closed_form_binary_logit_and_cluster_covariance(self):
        ll, scores, info, probability = likelihood_parts(self.beta, self.data)
        self.assertAlmostEqual(ll, 6*np.log(.75)+2*np.log(.25), places=13)
        np.testing.assert_allclose(probability, np.tile([.75, .25], (8, 1)), atol=1e-14)
        self.assertAlmostEqual(scores.sum(), 0, places=13)
        self.assertAlmostEqual(info[0, 0], 1.5, places=13)
        result = covariance(self.beta, self.data)
        self.assertAlmostEqual(result['conventional'][0, 0], 2/3, places=13)
        self.assertAlmostEqual(result['row_sandwich'][0, 0], 2/3, places=13)
        self.assertAlmostEqual(result['cluster_uncorrected'][0, 0], 4/3, places=13)
        self.assertAlmostEqual(result['cluster_corrected'][0, 0], 16/9, places=13)

    def test_fit_reaches_known_optimum(self):
        b, diagnostic = fit(self.data)
        np.testing.assert_allclose(b, self.beta, atol=1e-8)
        self.assertLessEqual(diagnostic['gradient_infinity_norm'], 1e-6)

    def test_gradient_and_information_against_finite_differences(self):
        rng = np.random.default_rng(731)
        x = rng.normal(size=(15, 3, 3))
        av = np.ones((15, 3)); av[::2, 2] = 0
        y = rng.integers(0, 2, 15)
        d = validate(x, av, y, np.arange(15))
        b = np.array([.2, -.3, .1]); step = 1e-5
        ll, score, info, _ = likelihood_parts(b, d)
        for j in range(3):
            shift = np.eye(3)[j] * step
            plus = likelihood_parts(b+shift, d)
            minus = likelihood_parts(b-shift, d)
            self.assertAlmostEqual((plus[0]-minus[0])/(2*step), score.sum(axis=0)[j], places=7)
            np.testing.assert_allclose(-(plus[1].sum(axis=0)-minus[1].sum(axis=0))/(2*step), info[:, j], atol=1e-8)

    def test_row_and_group_label_permutations(self):
        order = np.array([7, 3, 1, 6, 4, 2, 0, 5])
        other = validate(self.x[order], self.av[order], self.choice[order], (self.group[order]+17)*3)
        for key in ['conventional', 'row_sandwich', 'cluster_uncorrected', 'cluster_corrected']:
            np.testing.assert_allclose(covariance(self.beta, other)[key], covariance(self.beta, self.data)[key], atol=1e-14)

    def test_common_utility_shift_does_not_change_probabilities_or_information(self):
        shifted = self.x + 170
        other = validate(shifted, self.av, self.choice, self.group)
        a = likelihood_parts(self.beta, self.data); b = likelihood_parts(self.beta, other)
        np.testing.assert_allclose(a[3], b[3], atol=1e-13)
        np.testing.assert_allclose(a[1], b[1], atol=1e-13)
        np.testing.assert_allclose(a[2], b[2], atol=1e-13)

    def test_single_available_choice_has_zero_score_and_information(self):
        av = self.av.copy(); av[:, 1] = 0
        data = validate(self.x, av, np.zeros(8), self.group)
        ll, scores, information, _ = likelihood_parts(self.beta, data)
        self.assertEqual(ll, 0)
        np.testing.assert_array_equal(scores, np.zeros_like(scores))
        np.testing.assert_array_equal(information, np.zeros_like(information))
        with self.assertRaisesRegex(ValueError, 'singular'): covariance(self.beta, data)

    def test_choice_and_availability_refusals(self):
        for y in [self.choice[:-1], np.full(8, -1), np.full(8, 2), np.full(8, .5), np.ones(8, dtype=bool)]:
            with self.subTest(choice=y):
                with self.assertRaises(ValueError): validate(self.x, self.av, y, self.group)
        for av in [np.zeros_like(self.av), np.full_like(self.av, 2), np.ones((7, 2)), np.full((8, 2), np.nan)]:
            with self.subTest(availability=av):
                with self.assertRaises(ValueError): validate(self.x, av, self.choice, self.group)

    def test_finite_shape_and_group_refusals(self):
        cases = [(np.full_like(self.x, np.nan), self.group), (self.x.reshape(8, 2), self.group),
                 (self.x, self.group[:-1]), (self.x, np.full(8, np.nan)),
                 (self.x, np.full(8, np.nan, dtype=object)), (self.x, np.full(8, None))]
        for x, group in cases:
            with self.subTest(shape=x.shape):
                with self.assertRaises(ValueError): validate(x, self.av, self.choice, group)
        with self.assertRaisesRegex(ValueError, 'two groups'):
            covariance(self.beta, validate(self.x, self.av, self.choice, np.zeros(8)))

    def test_input_arrays_are_not_mutated(self):
        original = [x.copy() for x in [self.x, self.av, self.choice, self.group]]
        d = validate(self.x, self.av, self.choice, self.group)
        likelihood_parts(self.beta, d); covariance(self.beta, d); fit(d)
        for before, after in zip(original, [self.x, self.av, self.choice, self.group]):
            np.testing.assert_array_equal(before, after)

    def test_log_domain_keeps_extreme_unlikely_choice_finite(self):
        ll, _, _, p = likelihood_parts(np.array([1000.0]), self.data)
        self.assertTrue(np.isfinite(ll))
        self.assertAlmostEqual(ll, -2000, places=8)
        self.assertTrue(np.isfinite(p).all())
        with self.assertRaises(ValueError): likelihood_parts(np.array([np.nan]), self.data)
        with self.assertRaises(ValueError): likelihood_parts(np.array([1,2]), self.data)

    def test_reference_comparator_rejects_type_count_and_schema_changes(self):
        expected = {'count':8,'flag':True,'value':1.0,'names':['a','b']}
        compare_reference({'count':8,'flag':True,'value':1.0+1e-10,'names':['a','b']},expected)
        for change in [{'count':9},{'count':8.0},{'flag':1},{'value':float('nan')},
                       {'value':1.0001},{'names':['a']},{'names':['a','c']}]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError): compare_reference({**expected,**change},expected)
        with self.assertRaises(ValueError):compare_reference({**expected,'extra':0},expected)

    def test_registered_data_hash_refuses_corrupt_input_before_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.dat';path.write_bytes(b'bad')
            receipt={'raw_bytes':3,'raw_sha256':'0'*64}
            with self.assertRaisesRegex(ValueError,'input bytes'):acquire_input(path,receipt)

    def test_output_guard_refuses_existing_and_historical_paths_before_network(self):
        with self.assertRaisesRegex(ValueError,'separate'):run(None,ROOT/'results')
        with self.assertRaisesRegex(ValueError,'separate'):run(None,ROOT/'results'/'child')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileExistsError,'overwrite'):run(None,folder)


if __name__ == '__main__':
    unittest.main()
