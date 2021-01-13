from rec.base import ParametrizedObject
from rec.dataset.dataset import Dataset


class TestCasesGenerator(ParametrizedObject):
    def generate(self, dataset):
        """
        Test Cases generator for Session-Aware Recommendations evaluation.

        Different approaches requires slightly different test case preparation from
        users session. This class is base for every test case preparation for SARS
        scripts.

        :param dataset: test data set from which test cases should be generated
        :return:
         test_sessions : list of test sessions
         ground_truth : list of list of VIEWed items
        """
        assert isinstance(dataset, Dataset)
        pass


class AllViewedItemsTestCaseGenerator(TestCasesGenerator):
    def __init__(self):
        super(AllViewedItemsTestCaseGenerator, self).__init__()

    def generate(self, dataset):
        super(AllViewedItemsTestCaseGenerator, self).generate(dataset)
        sessions = [s for sessions in dataset.sessions.values() for s in sessions.values() if s.events_num() > 0]
        test_sessions = []
        ground_truth = []
        for s in sessions:
            relevant_items = s.clicked_items_set()
            if len(relevant_items) > 0:
                test_sessions.append(s)
                ground_truth.append(list(relevant_items))

        return test_sessions, ground_truth


class LeftNEventsTestCaseGenerator(TestCasesGenerator):
    def __init__(self, n=1, event_type=None):
        super(LeftNEventsTestCaseGenerator, self).__init__()
        self.n = n
        self.event_type = event_type

    def generate(self, dataset):
        super(LeftNEventsTestCaseGenerator, self).generate(dataset)

        # split test sessions
        test_eval_sessions = [
            s.split_session(self.event_type, -self.n) for sessions in list(dataset.sessions.values())
            for s in list(sessions.values())
        ]

        # test sessions = prediction events + evaluation
        test_sessions = []
        ground_truth = []
        for session_test, session_eval in test_eval_sessions:
            if session_test.events_num() > 0 and session_eval.events_num() > 0:
                relevant_items = session_eval.clicked_items_set()
                if len(relevant_items) > 0:
                    test_sessions.append(session_test)
                    ground_truth.append(list(relevant_items))

        return test_sessions, ground_truth


class SubsequentEventTestCaseGenerator(TestCasesGenerator):
    def __init__(self, only_new=True, positive_event_types=['VIEW']):
        super(SubsequentEventTestCaseGenerator, self).__init__()
        self.only_new = only_new
        self.positive_event_types = positive_event_types
        self._positive_item_type_set = set(positive_event_types)

    def generate(self, dataset):
        super(SubsequentEventTestCaseGenerator, self).generate(dataset)

        test_sessions = []
        ground_truth = []

        for s in [s for sessions in list(dataset.sessions.values()) for s in list(sessions.values())]:
            for event_num in range(s.events_num() - 1):
                before, after = s.split_session(n=event_num)
                relevant_items = after.clicked_items_list(event_types=self._positive_item_type_set)
                already_clicked = before.clicked_items_set()
                if len(relevant_items) > 0:
                    if self.only_new and len(set(relevant_items) & already_clicked) > 0:
                        next_items = [item for item in relevant_items if item not in already_clicked]
                        if len(next_items) == 0:
                            continue
                    else:
                        next_items = relevant_items
                    test_sessions.append(before)
                    ground_truth.append(next_items)

        return test_sessions, ground_truth
