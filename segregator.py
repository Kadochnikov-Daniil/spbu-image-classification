from transformers import SegformerFeatureExtractor, SegformerForImageClassification, pipeline

def initialize():
    name = "nvidia/mit-b0"
    extractor = SegformerFeatureExtractor.from_pretrained(name)
    model = SegformerForImageClassification.from_pretrained(name)
    return pipeline("image-classification", model=model, feature_extractor=extractor)

def classify(self, image):
    return self(image)
