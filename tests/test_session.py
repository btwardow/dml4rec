from unittest import TestCase

from rec.model import Session, PRESENTED_ITEMS, EVENT_ITEM


class TestSession(TestCase):
    def test_create_and_add_event(self):
        # given
        s = Session(1)
        s.create_and_add_event(self.create_event(2, 'E1'))
        s.create_and_add_event(self.create_event(3, 'SEARCH'))
        s.create_and_add_event(self.create_event(1, 'E2'))

        # then
        self.assertListEqual([1, 2, 3], [e.timestamp for e in s.events], 'Events should be sorted by timestamp')

    def create_event(self, ts, event_type, presented_items=None, clicked_item=None):
        e = {'timestamp': ts, 'eventType': event_type}
        if presented_items:
            e[PRESENTED_ITEMS] = presented_items
        if clicked_item:
            e[EVENT_ITEM] = clicked_item
        return e

    def test_events_split_on_n(self):
        # given
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'A'))
        s.create_and_add_event(self.create_event(3, 'B'))
        s.create_and_add_event(self.create_event(2, 'C'))

        # when
        before, after = s._split_events(n=0)

        self.assertListEqual(['A'], [e.event_type for e in before])
        self.assertListEqual(['C', 'B'], [e.event_type for e in after])

    def test_events_split_on_event_type(self):
        # given
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'A'))
        s.create_and_add_event(self.create_event(3, 'B'))
        s.create_and_add_event(self.create_event(2, 'C'))

        # when
        before, after = s._split_events('C')

        self.assertListEqual(
            ['A', 'C', 'B'], [e.event_type for e in before + after], 'Simple event split based on timestamp order'
        )

    def test_get_actions_after_last_event(self):
        # given
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'CLICK1'))
        s.create_and_add_event(self.create_event(2, 'SEARCH'))
        s.create_and_add_event(self.create_event(3, 'CLICK2'))
        s.create_and_add_event(self.create_event(4, 'CLICK3'))

        # when
        before, after = s._split_events('SEARCH', -1)

        self.assertListEqual(
            ['CLICK1', 'SEARCH'], [e.event_type for e in before],
            'Should return only events before SEARCH (including it!)'
        )
        self.assertListEqual(
            ['CLICK2', 'CLICK3'], [e.event_type for e in after], 'Should return events after last SEARCH'
        )

    def test_session_split(self):
        # given
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'CLICK1'))
        s.create_and_add_event(self.create_event(2, 'SEARCH'))
        s.create_and_add_event(self.create_event(3, 'CLICK2'))

        # when
        b, a = s.split_session(n=1)

        self.assertIsInstance(b, Session)
        self.assertIsInstance(a, Session)
        self.assertListEqual([1, 2], [e.timestamp for e in b.events])
        self.assertListEqual([3], [e.timestamp for e in a.events])

    def test_all_presented_items(self):
        # given
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'CLICK1', presented_items=[1, 2, 3]))
        s.create_and_add_event(self.create_event(2, 'SEARCH', presented_items=[1, 3, 4]))
        s.create_and_add_event(self.create_event(3, 'CLICK2'))
        s.create_and_add_event(self.create_event(4, 'CLICK3'))

        # when
        result = s.all_presented_items()

        # then
        self.assertSetEqual(result, set([1, 2, 3, 4]))

    def test_all_clicked_items_set(self):
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'CLICK1', presented_items=[1, 2, 3]))
        s.create_and_add_event(self.create_event(2, 'SEARCH', presented_items=[1, 3, 4], clicked_item=2))
        s.create_and_add_event(self.create_event(3, 'CLICK2'))
        s.create_and_add_event(self.create_event(4, 'CLICK3', clicked_item=3))

        # when
        result = s.clicked_items_set()

        # then
        self.assertSetEqual(result, set([2, 3]))

    def test_all_clicked_items_list(self):
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'CLICK1', presented_items=[1, 2, 3]))
        s.create_and_add_event(self.create_event(2, 'SEARCH', presented_items=[1, 3, 4], clicked_item=2))
        s.create_and_add_event(self.create_event(3, 'CLICK2', clicked_item=2))
        s.create_and_add_event(self.create_event(4, 'CLICK3', clicked_item=3))

        # when
        result = s.clicked_items_list()

        # then
        self.assertListEqual(result, [2, 2, 3])

    def test_is_in_session(self):
        # given
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'A', clicked_item=1))
        s.create_and_add_event(self.create_event(3, 'B', clicked_item=2))
        s.create_and_add_event(self.create_event(2, 'C', clicked_item=3))

        # when and then
        self.assertTrue(s.is_in_session('A', 1))
        self.assertFalse(s.is_in_session('C', 1))

    def test_presented_not_clicked(self):
        s = Session(1)
        s.create_and_add_event(self.create_event(1, 'CLICK1', presented_items=[1, 2, 3]))
        s.create_and_add_event(self.create_event(2, 'SEARCH', presented_items=[1, 3, 4], clicked_item=2))
        s.create_and_add_event(self.create_event(3, 'CLICK2', clicked_item=2))
        s.create_and_add_event(self.create_event(4, 'CLICK3', clicked_item=3))

        # when
        result = s.presented_not_clicked()

        # then
        self.assertSetEqual(result, {1, 4})
