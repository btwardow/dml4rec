"""
Module with the main model used in recommendations.
"""

from collections import namedtuple

# event fields
TIMESTAMP = 'timestamp'
PRESENTED_ITEMS = 'presentedItems'
EVENT_ITEM = 'clickedItem'
EVENT_TYPE = 'eventType'
EVENT_CONTEXT = 'context'
EVENT_USER_HASH = 'userHash'
EVENT_USER_ID = 'userId'
EVENT_SESSION_ID = 'sessionId'

# item fields
ITEM_ID = 'id'


class Session(object):
    def __init__(self, id, user_hash=None, user_id=None):
        """
        Creates object representing User's Session. 
        
        Session consists of the user's sequence of events, where each event represent
         a particular user action, e.g. CLICK, BUY, SEARCH. The sequence is ordered 
         by timestamp. The base field of the event and session are present. The additional
         data of the event should be placed in the context dict. 
        
        Args:
            id: unique session id
            user_hash: (optional) user hash (e.g. cookie, fingerprint) 
            user_id:  (optional) user numeric id if available
        """
        self.id = id
        self.user_hash = user_hash
        self.user_id = user_id

        self.events = []
        self.timestamp_start = None
        self.timestamp_end = None

    def create_and_add_event(self, event_data):
        """
        Create and event from the dictionary and place it to the 
         session.
        Args:
            event_data: (dict) event data 

        Returns:

        """
        event = event_from_dict(event_data)
        self.add_event(event)

    def add_event(self, event):
        self.events.append(event)

        if self.timestamp_start is None or self.timestamp_start > event.timestamp:
            self.timestamp_start = event.timestamp

        if self.timestamp_end is None or self.timestamp_end < event.timestamp:
            self.timestamp_end = event.timestamp

        # when event was place somewhere between
        # TODO: This one is not the most efficient way of doing this
        if self.timestamp_end != event.timestamp:
            self.events = sorted(self.events, key=lambda e: e.timestamp)

    def _split_events(self, event_type=None, n=-1):
        if event_type is None:
            return self.events[:n + 1], self.events[n + 1:]
        else:
            event_idx = len(self.events)
            occurrence = abs(n)
            rev = n < 0

            for i in range(0, len(self.events)):
                if self.events[-i if rev else i].event_type == event_type:
                    occurrence -= 1
                    event_idx = i
                    if occurrence == 0:
                        break
            if rev:
                event_idx = len(self.events) - event_idx

            return self.events[:event_idx + 1], self.events[event_idx + 1:]

    def split_session(self, event_type=None, n=-1):
        before, after = self._split_events(event_type, n)
        session_before = Session(self.id, self.user_hash, self.user_id)
        [session_before.add_event(e) for e in before]
        session_after = Session(self.id, self.user_hash, self.user_id)
        [session_after.add_event(e) for e in after]
        return session_before, session_after

    def all_presented_items(self):
        presented_items = set()
        for e in self.events:
            if e.presented_items:
                presented_items |= set(e.presented_items)
            if e.clicked_item:
                presented_items |= {e.clicked_item}

        return presented_items

    def presented_not_clicked(self):
        return self.all_presented_items() - self.clicked_items_set()

    def all_items(self):
        all_items = set()
        for e in self.events:
            if e.presented_items:
                all_items |= set(e.presented_items)
            if e.clicked_item:
                all_items.add(e.clicked_item)
        return all_items

    def clicked_items_set(self, event_types=None):
        clicked_items = set()
        for e in self.events:
            if e.clicked_item and (event_types is None or e.event_type in event_types):
                clicked_items.add(e.clicked_item)
        return clicked_items

    def clicked_items_list(self, event_types=None):
        """
        Return clicked item list.
        Args:
            event_types: optional item types to consider.

        Returns:

        """
        clicked_items = list()
        for e in self.events:
            if e.clicked_item and (event_types is None or e.event_type in event_types):
                clicked_items.append(e.clicked_item)
        return clicked_items

    def events_num(self):
        return len(self.events)

    def is_in_session(self, event_type, item_id):
        """
        Check if the given event type and item is in the session.
        Args:
            event_type: 
            item_id: 

        Returns: bool

        """
        for e in self.events:
            if e.event_type == event_type and e.clicked_item == item_id:
                return True
        return False

    def __len__(self):
        return len(self.events)


Event = namedtuple('Event', ['timestamp', 'event_type', 'clicked_item', 'presented_items', 'context'])
Event.__new__.__defaults__ = (None, ) * (len(Event._fields) - 3)  # only id is required


def event_from_dict(event_data):
    """
    Creates Event object from dictionary.
    Timestamp in event can be in seconds or milliseconds. Milliseconds will be truncated to sec.

    Args:
        event_data (dict): dictionary with event data.

    Returns:
        event (Event): Event object (namedtuple)
    """
    ts = event_data[TIMESTAMP]
    # normalize to seconds
    if ts > 1000000000000:
        ts /= 1000

    clicked_item = event_data[EVENT_ITEM] if EVENT_ITEM in event_data else None
    presented_items = event_data[PRESENTED_ITEMS] if PRESENTED_ITEMS in event_data else None
    context = event_data[EVENT_CONTEXT] if EVENT_CONTEXT in event_data else None
    return Event(
        timestamp=ts,
        event_type=event_data[EVENT_TYPE],
        clicked_item=clicked_item,
        presented_items=presented_items,
        context=context
    )


Item = namedtuple('Item', ['id', 'data'])
Item.__new__.__defaults__ = (None, ) * (len(Item._fields) - 1)  # only id is required
