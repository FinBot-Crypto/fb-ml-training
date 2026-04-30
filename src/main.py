import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-training-service")

def main():
    logger.info("Service ml-training-service starting...")
    while True:
        logger.info("Service ml-training-service is running...")
        time.sleep(60)

if __name__ == "__main__":
    main()
