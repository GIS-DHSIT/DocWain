from typing import List, Optional
from django.core.files.uploadedfile import UploadedFile
from apps.core.services.ingestor import Ingestor
from apps.core.services.chain import create_chain
from apps.core.services.processor import create_llm
from apps.core.services.retriever import create_retriever
from .models import Document, Message, Conversation
from apps.core.utils.logger import setup_logger

class DocumentService:
    @staticmethod
    async def process_documents(files: List[UploadedFile]) -> bool:
        """Process uploaded documents"""
        try:
            # Initialize ingestor
            ingestor = Ingestor()

            # Process documents and create vector store
            vector_store = await ingestor.ingest(files)

            # Create retrieval chain
            llm = create_llm()
            retriever = create_retriever(llm, vector_store)
            chain = create_chain(llm, retriever)

            # Store chain in session or cache
            # TODO: Implement proper chain storage/retrieval

            return True
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False


class ChatService:
    @staticmethod
    async def get_response(question: str, conversation_id: Optional[int] = None) -> str:
        """Get response from the chat chain"""
        try:
            # Get or create conversation
            if conversation_id:
                conversation = await Conversation.objects.aget(id=conversation_id)
            else:
                conversation = await Conversation.objects.acreate()

            # Create message
            await Message.objects.acreate(
                conversation=conversation,
                role="user",
                content=question
            )

            # Get chain from session/cache
            # TODO: Implement proper chain retrieval
            chain = None  # Placeholder

            if not chain:
                return "Please upload documents first."

            # Get response
            response = await chain.ainvoke({"question": question})

            # Save assistant message
            await Message.objects.acreate(
                conversation=conversation,
                role="assistant",
                content=response
            )

            return response
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return f"Error: {str(e)}"