import time

class Simulator:
    
    def __init__(self,
                 starting_state,
                 environment_model,
                 event_processing_callback,
                 starting_event_queue,
                 logger) -> None:
        
        self.state = starting_state
        self.environment_model = environment_model
        self.event_processing_callback = event_processing_callback
        self.event_queue = starting_event_queue
        self.logger = logger
        
        self.start_sim_time = None
        self.starting_num_events = len(starting_event_queue)
        self.num_events_processed = 0
        
    def run_simulation(self):
        self.logger.info("Running simulation.")
        
        self.start_sim_time = time.time()
        
        while len(self.event_queue) > 0:
            self.update_sim_info()
            pass
        
        
    def update_sim_info(self):
        self.num_events_processed += 1