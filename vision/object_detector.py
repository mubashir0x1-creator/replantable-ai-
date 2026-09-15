class ObjectDetector:
    def __init__(self):
        self.known_objects = ["plate", "cup"]

    def detect(self, objects):
        """
        Detect known objects from a list of object names.

        Example:
            detector.detect(["plate", "cup"])
        """
        detections = []

        for object_name in objects:
            if object_name in self.known_objects:
                detections.append({
                    "object": object_name,
                    "detected": True,
                })

        return detections


if __name__ == "__main__":
    detector = ObjectDetector()

    result = detector.detect(["plate", "cup"])

    print("=" * 64)
    print("VISION OBJECT DETECTION")
    print("=" * 64)

    for item in result:
        print(f"{item['object']}: detected={item['detected']}")

    print("=" * 64)
