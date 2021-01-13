import collections

from rec.dataset.dataset import Dataset
from rec.model import Session, TIMESTAMP, EVENT_ITEM, EVENT_TYPE


class SessionClock:
    def __init__(self) -> None:
        self.session_clock = 0

    def tick(self):
        self.session_clock += 1
        return self.session_clock


default_session_clock = SessionClock()


def create_event(timestamp, item_id):
    return {TIMESTAMP: timestamp, EVENT_ITEM: item_id, EVENT_TYPE: 'CLICK'}


def create_session(session_id, *items, clock: SessionClock = default_session_clock):
    s = Session(session_id)
    for i in items:
        s.create_and_add_event(create_event(clock.tick(), i))
    return s


def wrap_sessions(*sessions, user_id="u1"):
    user_sessions = collections.OrderedDict([(s.id, s) for s in sessions])
    return {user_id: user_sessions}


def create_dataset(*sessions):
    max_item_id = max([event.clicked_item for session in sessions for event in session.events])

    dataset = Dataset()
    dataset.items = collections.OrderedDict([(item_id, {}) for item_id in range(max_item_id + 1)])

    dataset.sessions = wrap_sessions(*sessions)
    dataset._create_indexes()
    return dataset
