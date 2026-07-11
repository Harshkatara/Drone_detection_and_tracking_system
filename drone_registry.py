import time

class DroneRegistry:

    def __init__(self):

        self.registry = {}
        self.track_to_drone = {}
        self.next_id = 1

    def get_drone_id(
        self,
        track_id,
        center_x,
        center_y,
        box_width,
        box_height
    ):

        if track_id in self.track_to_drone:

            drone_id = self.track_to_drone[track_id]

            self.registry[drone_id]["last_seen"] = time.time()

            self.registry[drone_id]["x"] = center_x
            self.registry[drone_id]["y"] = center_y

            return drone_id

        best_match = None
        best_distance = 999999

        for drone_id, data in self.registry.items():

            dx = center_x - data["x"]
            dy = center_y - data["y"]

            distance = (dx * dx + dy * dy) ** 0.5

            if distance < 100:

                if distance < best_distance:

                    best_distance = distance
                    best_match = drone_id

        # if best_match:

        #     self.track_to_drone[track_id] = best_match

        #     self.registry[best_match]["last_seen"] = time.time()

        #     return best_match

        if best_match:

            self.track_to_drone[track_id] = best_match

            self.registry[best_match]["last_seen"] = time.time()

            self.registry[best_match]["x"] = center_x
            self.registry[best_match]["y"] = center_y

            return best_match

        drone_id = f"DRONE_{self.next_id:03d}"

        self.next_id += 1

        self.registry[drone_id] = {
            "x": center_x,
            "y": center_y,
            "width": box_width,
            "height": box_height,
            "last_seen": time.time()
        }

        self.track_to_drone[track_id] = drone_id

        return drone_id