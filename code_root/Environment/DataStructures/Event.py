class Event:
    
    def __init__(self, 
                 event_type,
                 time,
                 type_specific_information=None):
        self.event_type = event_type
        self.time = time
        self.type_specific_information = type_specific_information