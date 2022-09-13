
    
class DoNothing:
    
    def __init__(self, 
                 environment_model,
                 travel_model,
                 dispatch_policy) -> None:
        self.environment_model = environment_model
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.event_couter = 0
        
        pass
    
    def event_processing_callback_funct(self, state, curr_event, next_event):
        
        self.event_couter += 1
        pass