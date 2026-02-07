
import logging
import numpy as np
from src.api.config import Config
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, InputExample
from transformers import AdamW, get_linear_schedule_with_warmup

# Initialize Qdrant client and model
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)

def retrieve_embeddings_from_qdrant(tag):
    """Retrieve embeddings from Qdrant for a specific tag."""
    try:
        search_result = qdrant_client.scroll(
            collection_name=tag,
            scroll_filter=Filter(must=[{"key": "tag", "match": {"value": tag}}]),
            limit=1000,
            with_vectors=True
        )

        if not search_result or not search_result[0]:
            logging.warning("No relevant data found in Qdrant.")
            return None

        from src.utils.payload_utils import get_canonical_text

        texts = [get_canonical_text(point.payload or {}) for point in search_result[0]]
        texts = [text for text in texts if text]
        embeddings = np.array([point.vector for point in search_result[0] if point.vector], dtype=np.float32)

        if embeddings.size == 0:
            logging.warning("Embeddings found but vectors are missing!")
            return None

        return texts, embeddings

    except Exception as e:
        logging.error(f"Error retrieving embeddings from Qdrant: {e}")
        return None

def fine_tune_model(tag, batch_size=32, epochs=3, learning_rate=2e-5):
    """Fine-tune the model using embeddings from Qdrant."""
    try:
        logging.info(f"Starting fine-tuning process for tag: {tag}")
        data = retrieve_embeddings_from_qdrant(tag)
        if not data:
            return "No data found for fine-tuning."

        texts, embeddings = data

        # Prepare training data
        train_examples = [InputExample(texts=[text], label=embedding) for text, embedding in zip(texts, embeddings)]
        train_dataset = SentencesDataset(train_examples, MODEL)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        # Define loss function
        train_loss = losses.CosineSimilarityLoss(MODEL)

        # Optimizer and learning rate scheduler
        optimizer = AdamW(MODEL.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Training loop
        MODEL.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            show_progress_bar=True
        )

        # Save the updated model
        MODEL.save('/home/muthu/PycharmProjects/DocWain/model_training')
        logging.info("Model fine-tuning completed and saved")
        return "Fine-tuning completed successfully."

    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        return "Fine-tuning failed."


def retrieve_tags():
    """Retrieve all unique tags from Qdrant."""
    try:
        unique_tags = set()
        scroll_filter = Filter(must=[])
        scroll_result = qdrant_client.scroll(
            collection_name='your_collection_name',
            scroll_filter=scroll_filter,
            limit=1000,
            with_payload=True
        )

        while scroll_result:
            for point in scroll_result[0]:
                if 'tag' in point.payload:
                    unique_tags.add(point.payload['tag'])
            scroll_result = qdrant_client.scroll(
                collection_name='your_collection_name',
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
                offset=scroll_result[1]
            )

        return list(unique_tags)

    except Exception as e:
        logging.error(f"Error retrieving tags from Qdrant: {e}")
        return []
# Example usage
tags = retrieve_tags()
for tag in tags:
    logging.info(f"Fine-tuning model for tag: {tag}")
    fine_tune_model(tag)
