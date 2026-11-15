class LapCompletion:

    def __init__(self, track_length, start_progress=0.0, crossing_threshold=0.10,):
        self.track_length = float(track_length)
        self.start_progress = float(start_progress)
        self.crossing_threshold = float(crossing_threshold)

        if self.track_length <= 0.0:
            raise ValueError("track_length must be positive.")

        if not 0.0 < self.crossing_threshold < 0.5:
            raise ValueError("crossing_threshold must be between 0 and 0.5.")

        self.lap_count = 0
        self.reset()

    def reset(self):
        self.lap_count = 0

    def _relative_progress(self, progress):
        return (float(progress) - self.start_progress) % self.track_length

    def update(self, previous_progress, current_progress, progress_delta):
        previous_progress = float(previous_progress)
        current_progress = float(current_progress)
        progress_delta = float(progress_delta)

        if progress_delta <= 0.0:
            return False

        previous_relative = self._relative_progress(previous_progress)
        current_relative = self._relative_progress(current_progress)

        start_zone = self.track_length * self.crossing_threshold
        finish_zone = self.track_length * (1.0 - self.crossing_threshold)

        crossed_finish_line = (previous_relative >= finish_zone and current_relative <= start_zone)

        if crossed_finish_line:
            self.lap_count += 1
            return True

        return False

    def get_lap_count(self):
        return int(self.lap_count)