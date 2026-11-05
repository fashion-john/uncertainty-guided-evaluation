"""Small semantic checks for the published metric and perturbation definitions."""
import unittest

from protocol import distribution, make_views, measurements, parse_label, position_map, record
from reproduce import tables


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.base = {'question': 'Example?', 'options': dict(zip('ABCD', ['one', 'two', 'three', 'four'])), 'answer': 'A'}

    def test_unique_label_parser(self):
        self.assertEqual(parse_label('Final answer: (B)'), 'B')
        self.assertEqual(parse_label('A or B'), '')
        self.assertEqual(parse_label('', ('I', 'II', 'III', 'IV')), '')
        self.assertEqual(parse_label('III', ('I', 'II', 'III', 'IV')), 'III')

    def test_medical_swap_and_existing_D(self):
        views = make_views('medqa', '0000', self.base)
        self.assertEqual(position_map('medqa', '0000', self.base), 'DBCA')
        self.assertEqual(views['fixed_pos']['options']['D'], 'one')
        self.assertEqual(views['fixed_pos']['answer'], 'D')
        self.assertEqual(position_map('medqa', '0000', {**self.base, 'answer': 'D'}), 'ABCD')

    def test_duplicate_answer_text_preserves_historical_swap(self):
        base = {**self.base, 'answer': 'B', 'options': dict(zip('ABCD', ['same', 'same', 'third', 'fourth']))}
        self.assertEqual(position_map('medmcqa', '0000', base), 'DBCA')
        self.assertEqual(make_views('medmcqa', '0000', base)['fixed_pos']['answer'], 'D')

    def test_openbook_permutation(self):
        mapping = position_map('openbookqa', 'openbookqa-00000', self.base)
        self.assertEqual(set(mapping), set('ABCD'))
        self.assertNotEqual(mapping, 'ABCD')
        self.assertEqual(mapping, position_map('openbookqa', 'openbookqa-00000', self.base))

    def test_hidden_error_requires_original_correct(self):
        views = make_views('medqa', '0000', self.base)
        probabilities = dict(zip('ABCD', [.4, .3, .2, .1]))
        row = record('Gemini-2.5-Flash', 'medqa', '0000', 0, views,
                     dict(zip(('mcq', 'roman_numeral', 'fixed_pos'), ('A', 'II', 'D'))), probabilities, False)
        value = measurements(row)
        self.assertEqual((value['hidden'], value['strict'], value['label_pair'], value['position_pair']), (1, 0, 0, 1))
        row['pred_mcq'] = 'B'
        self.assertEqual(measurements(row)['hidden'], 0)

    def test_consistent_wrong_answers_are_consistent(self):
        views = make_views('medqa', '0000', self.base)
        row = record('Gemini-2.5-Flash', 'medqa', '0000', 0, views,
                     dict(zip(('mcq', 'roman_numeral', 'fixed_pos'), ('B', 'II', 'B'))), {}, False)
        self.assertEqual((measurements(row)['consistent'], measurements(row)['strict']), (1, 0))

    def test_missing_probability_and_label_token_aggregation(self):
        p, imputed = distribution([{'token': 'A', 'logProbability': -1, 'tokenId': 1},
                                  {'token': ' A', 'logProbability': -2, 'tokenId': 2},
                                  {'token': 'B', 'logProbability': -3, 'tokenId': 3}])
        self.assertTrue(imputed)
        self.assertAlmostEqual(sum(p.values()), 1)
        self.assertGreater(p['A'], p['B'])
        self.assertEqual(distribution([]), ({}, False))

    def test_ties_follow_source_order_and_error_matches_recall(self):
        rows = []
        probabilities = dict(zip('ABCD', [.4, .3, .2, .1]))
        for index in range(500):
            views = make_views('openbookqa', str(index), self.base)
            answers = {mode: item['answer'] for mode, item in views.items()}
            if 140 <= index < 150:
                gold = answers['fixed_pos']
                answers['fixed_pos'] = next(lab for lab in 'ABCD' if lab != gold)
            rows.append(record('Gemini-2.5-Flash', 'openbookqa', str(index), index,
                               views, answers, probabilities, False))
        results = tables(list(reversed(rows)))
        # All scores are tied. At 30% coverage source positions 0..149 are selected.
        for row in results['table54']:
            self.assertEqual(row['hidden_recall30_pct'], '100.00')
            self.assertEqual(row['error30_pp'], '0.000')
            self.assertEqual(row['lift10'], '0.000')
        self.assertEqual(results['table53'][0]['error20_pp'], '2.000')
        self.assertEqual(results['table52'][0]['recall30_pct'], '')
        with self.assertRaises(ValueError):
            tables(rows[:-1])


if __name__ == '__main__':
    unittest.main()
