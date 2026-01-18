import foxglove
import time

from foxglove.schemas import Log, LogLevel, Timestamp

server = foxglove.start_server()

while True:
    foxglove.log(
        "/hello",
        Log(
            timestamp=Timestamp.now(),
            level=LogLevel.Info,
            message="Hello, Foxglove!",
        )
    )

    time.sleep(0.033)
