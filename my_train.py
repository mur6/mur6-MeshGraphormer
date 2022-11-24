from my_model_tools import get_model
from logging import INFO, basicConfig, getLogger

logger = getLogger(__name__)

def main():
    device = "cpu"
    model = get_model(device)
    print(model)
    logger.info("Model Loaded.")

if __name__ == "__main__":
    basicConfig(level=INFO)
    main()


