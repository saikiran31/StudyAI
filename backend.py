from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import os
import json
import fitz
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from datetime import datetime
import time
import logging
import numpy as np

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],  # Add your frontend URL
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

app.config.update(
    SECRET_KEY='your-secret-key',
    SQLALCHEMY_DATABASE_URI='sqlite:///studyai.db',
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024
)
app.config['GROQ_API_KEY'] = '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'  # Replace with your actual API key
groq_client = Groq(api_key=app.config['GROQ_API_KEY'])
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

db = SQLAlchemy(app)

# Modified Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    documents = db.relationship('Document', backref='user', lazy=True)
    quizzes = db.relationship('Quiz', backref='user', lazy=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    file_path = db.Column(db.String(500))  # Store PDF file path
    embeddings = db.Column(db.Text)  # Store embeddings as JSON
    chunks = db.Column(db.Text)  # Store text chunks as JSON
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    questions = db.Column(db.JSON)  # Store questions as JSON
    score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    difficulty = db.Column(db.String(20), default='medium')

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    title = db.Column(db.String(200))
    messages = db.relationship('ChatMessage', backref='chat', lazy=True)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StudyAI:
    def __init__(self, groq_api_key: str):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.groq_client = Groq(api_key=groq_api_key)

    def process_document(self, file_path: str, user_id: int) -> Document:
        start_time = time.time()
        
        # Extract text
        text = self._extract_text(file_path)
        
        # Create chunks
        chunks = self._create_chunks(text)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        # Store in FAISS index
        self.index.add(embeddings)
        
        # Create document record
        document = Document(
            title=os.path.basename(file_path),
            content=text,
            chunk_data=json.dumps(chunks),
            embedding_data=json.dumps(embeddings.tolist()),
            user_id=user_id
        )
        
        db.session.add(document)
        db.session.commit()
        
        processing_time = time.time() - start_time
        if processing_time > 10:
            logging.warning(f"Document processing took {processing_time}s")
            
        return document

    def _extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = ""
            try:
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                return text
            except Exception as e:
                logging.error(f"PDF extraction error: {str(e)}")
                raise
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                while end > start and text[end] != ' ':
                    end -= 1
            chunks.append(text[start:end])
            start = end - overlap
            
        return chunks

    def generate_quiz(self, document_id: int, num_questions: int = 5, difficulty: str = 'medium') -> Quiz:
        """Generate a quiz with optimized content handling."""
        start_time = time.time()
        
        try:
            document = Document.query.get_or_404(document_id)
            chunks = json.loads(document.chunk_data)
            
            # Extract key information from document
            content_summary = ' '.join(chunks[0].split('\n')[:10])  # Take first 10 lines
            
            # Create a more focused prompt with smaller content
            prompt = f"""Create a multiple choice quiz about the following text:

    TEXT:
    {content_summary}

    Create exactly {num_questions} questions following these rules:
    1. Each question must be about the text
    2. Each question must have exactly 4 options labeled A) through D)
    3. Include the correct answer and a brief explanation

    Respond with ONLY the following JSON format:
    {{
        "title": "Quiz Title",
        "questions": [
            {{
                "text": "Question here?",
                "options": [
                    "A) Option 1",
                    "B) Option 2",
                    "C) Option 3",
                    "D) Option 4"
                ],
                "correct_answer": "A) Option 1",
                "explanation": "Brief explanation here"
            }}
        ]
    }}
    """

            # Generate quiz with adjusted parameters
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quiz generator that outputs only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=4000,  # Increased token limit
                top_p=0.95
            )
            
            # Get raw response
            raw_response = response.choices[0].message.content.strip()
            logging.info(f"Raw response length: {len(raw_response)}")
            
            try:
                # Try to parse JSON
                quiz_data = json.loads(raw_response)
                
                # Validate structure
                if not isinstance(quiz_data, dict):
                    raise ValueError("Response is not a JSON object")
                if "title" not in quiz_data or "questions" not in quiz_data:
                    raise ValueError("Missing required fields")
                    
                # Validate questions
                for q in quiz_data["questions"]:
                    if not all(k in q for k in ["text", "options", "correct_answer", "explanation"]):
                        raise ValueError("Question missing required fields")
                    if len(q["options"]) != 4:
                        raise ValueError("Question does not have exactly 4 options")
                    
                # Create quiz
                quiz = Quiz(
                    title=quiz_data["title"],
                    document_id=document_id,
                    user_id=document.user_id,
                    questions=json.dumps(quiz_data["questions"]),
                    difficulty=difficulty
                )
                
                db.session.add(quiz)
                db.session.commit()
                
                generation_time = time.time() - start_time
                if generation_time > 13:
                    logging.warning(f"Quiz generation took {generation_time}s")
                
                return quiz
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {str(e)}")
                logging.error(f"Raw response: {raw_response}")
                raise ValueError("Failed to parse quiz response")
                
        except Exception as e:
            logging.error(f"Quiz generation error: {str(e)}")
            db.session.rollback()
            raise

    def _create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100):
        """Create smaller chunks to avoid token limits."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                # Find the last newline before chunk_size
                while end > start and text[end] != '\n':
                    end -= 1
            chunks.append(text[start:end].strip())
            start = end - overlap
            
        return chunks

    def _parse_json_response(self, response_text: str) -> dict:
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"JSON parsing error: {str(e)}")
            raise ValueError("Invalid response format")

# Initialize StudyAI
study_ai = StudyAI(groq_api_key=app.config['GROQ_API_KEY'])

class DocumentProcessor:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', groq_api_key=None):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.groq_client = Groq(api_key=groq_api_key)
        
    def process_document(self, file_path):
        """Process PDF document and create embeddings."""
        # Extract text
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Create chunks
        chunks = self._create_chunks(text)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        return text, chunks, embeddings.tolist()
    
    def _create_chunks(self, text):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            if end < text_len:
                while end > start and text[end] != ' ':
                    end -= 1
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
        return chunks
    
    def query_document(self, query, chunks, embeddings):
        """Query document using RAG with Groq."""
        # Convert embeddings back to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search for relevant chunks
        k = 3  # Number of chunks to retrieve
        D, I = index.search(query_embedding, k)
        
        # Get relevant context
        context = "\n".join([chunks[i] for i in I[0]])
        
        # Generate response using Groq
        prompt = f"""Answer the following question based ONLY on the provided context. 
If the question cannot be answered from the context, say "I cannot answer this question based on the provided document."

Context:
{context}

Question: {query}

Answer:"""
        
        response = self.groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content

# Initialize processor
document_processor = DocumentProcessor(groq_api_key=app.config['GROQ_API_KEY'])

# Update document endpoints
@app.route('/upload-document', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    user_id = request.form.get('user_id')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process document
            text, chunks, embeddings = document_processor.process_document(file_path)
            
            # Create document record
            document = Document(
                title=filename,
                content=text,
                file_path=file_path,
                chunks=json.dumps(chunks),
                embeddings=json.dumps(embeddings),
                user_id=user_id
            )
            
            db.session.add(document)
            db.session.commit()
            
            return jsonify({
                'message': 'Document processed successfully',
                'document_id': document.id
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/get-document/<int:document_id>', methods=['GET'])
def get_document(document_id):
    document = Document.query.get_or_404(document_id)
    return send_file(document.file_path)

@app.route('/query-document', methods=['POST'])
def query_document():
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        query = data.get('query')
        
        document = Document.query.get_or_404(document_id)
        chunks = json.loads(document.chunks)
        embeddings = json.loads(document.embeddings)
        
        response = document_processor.query_document(query, chunks, embeddings)
        
        return jsonify({
            'response': response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Flask Routes
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Missing username or password'}), 400
            
        user = User(
            username=data['username'],
            password=generate_password_hash(data['password'])
        )
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': user.id
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if user and check_password_hash(user.password, data['password']):
        return jsonify({
            'message': 'Login successful',
            'user_id': user.id
        })
    
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/get-analytics', methods=['GET'])
def get_analytics():
    try:
        user_id = request.headers.get('Authorization')
        if not user_id:
            return jsonify({'error': 'Authorization required'}), 401

        # Get user's quizzes
        quizzes = Quiz.query.filter_by(user_id=user_id).all()
        quiz_scores = [
            {
                'date': quiz.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'score': quiz.score
            }
            for quiz in quizzes if quiz.score is not None
        ]

        # Get document progress
        documents = Document.query.filter_by(user_id=user_id).all()
        doc_progress = [
            {
                'title': doc.title,
                'progress': calculate_document_progress(doc.id)  # Implement this function
            }
            for doc in documents
        ]

        # Get recent activity
        recent_activity = []
        
        # Add quiz activities
        for quiz in quizzes[-5:]:  # Last 5 quizzes
            recent_activity.append({
                'type': 'quiz',
                'description': f"Completed quiz for {quiz.title}",
                'timestamp': quiz.created_at.isoformat()
            })
        
        # Add document activities
        for doc in documents[-5:]:  # Last 5 documents
            recent_activity.append({
                'type': 'document',
                'description': f"Uploaded document {doc.title}",
                'timestamp': doc.upload_date.isoformat()
            })
        
        # Sort by timestamp
        recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
        recent_activity = recent_activity[:5]  # Keep only 5 most recent

        return jsonify({
            'quizScores': quiz_scores,
            'documentProgress': doc_progress,
            'recentActivity': recent_activity
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_document_progress(document_id):
    """Calculate document study progress based on multiple factors."""
    try:
        document = Document.query.get_or_404(document_id)
        
        # Get all quizzes for this document
        quizzes = Quiz.query.filter_by(document_id=document_id).all()
        
        # Base progress metrics
        metrics = {
            'has_document': 20,  # Base progress for having uploaded the document
            'quiz_completion': 0, # Up to 50% based on quiz completion and scores
            'interaction': 0,     # Up to 30% based on interactions
        }
        
        # Calculate quiz metrics
        if quizzes:
            quiz_scores = [q.score for q in quizzes if q.score is not None]
            if quiz_scores:
                avg_score = sum(quiz_scores) / len(quiz_scores)
                # Weight both completion and performance
                metrics['quiz_completion'] = min(50, (len(quiz_scores) * 10) * (avg_score / 100))
        
        # Calculate interaction score based on chunks read/queried
        # Assuming we track interaction counts in the database
        chunks = json.loads(document.chunks) if document.chunks else []
        if chunks:
            # Award points for percentage of document covered
            metrics['interaction'] = min(30, len(chunks) * 2)
        
        # Sum all progress metrics
        total_progress = sum(metrics.values())
        
        # Cap at 100%
        return min(100, round(total_progress))
        
    except Exception as e:
        print(f"Error calculating progress: {str(e)}")
        return 0

@app.route('/get-documents', methods=['GET', 'OPTIONS'])
def get_documents():
    if request.method == 'OPTIONS':
        return '', 204
        
    user_id = request.headers.get('Authorization')
    if not user_id:
        return jsonify({'error': 'Authorization required'}), 401
    
    documents = Document.query.filter_by(user_id=user_id).all()
    return jsonify([{
        'id': doc.id,
        'title': doc.title,
        'upload_date': doc.upload_date.isoformat()
    } for doc in documents])

@app.route('/delete-document/<int:document_id>', methods=['DELETE', 'OPTIONS'])
def delete_document(document_id):
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        user_id = request.headers.get('Authorization')
        if not user_id:
            return jsonify({'error': 'Authorization required'}), 401
        
        document = Document.query.get_or_404(document_id)
        if str(document.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized access'}), 403
            
        db.session.delete(document)
        db.session.commit()
        
        return jsonify({'message': 'Document deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete document: {str(e)}'}), 500

@app.route('/generate-quiz', methods=['POST', 'OPTIONS'])
def generate_quiz():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        num_questions = data.get('num_questions', 5)
        difficulty = data.get('difficulty', 'medium')
        user_id = request.headers.get('Authorization')

        # Get document content
        document = Document.query.get_or_404(document_id)
        
        # Create quiz prompt
        prompt = f"""Create a quiz with {num_questions} questions based on this content:

{document.content}

Create exactly {num_questions} questions following these rules:
1. Each question must be about the text
2. Each question must have exactly 4 options labeled A) through D)
3. Include the correct answer and a brief explanation
4. Difficulty level: {difficulty}

Respond with ONLY this JSON format:
{{
    "title": "Quiz Title",
    "questions": [
        {{
            "text": "Question text",
            "options": [
                "A) First option",
                "B) Second option",
                "C) Third option",
                "D) Fourth option"
            ],
            "correct_answer": "A) First option",
            "explanation": "Brief explanation"
        }}
    ]
}}"""

        # Generate quiz using Groq
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a quiz generator that outputs only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=4000
        )

        # Parse response
        quiz_data = json.loads(response.choices[0].message.content.strip())
        
        # Validate quiz structure
        if not isinstance(quiz_data, dict) or 'questions' not in quiz_data:
            raise ValueError("Invalid quiz format received")
        
        # Create quiz record
        quiz = Quiz(
            title=quiz_data['title'],
            document_id=document_id,
            user_id=user_id,
            questions=quiz_data['questions'],
            difficulty=difficulty
        )
        db.session.add(quiz)
        db.session.commit()
        
        return jsonify({
            'quiz_id': quiz.id,
            'title': quiz.title,
            'questions': quiz_data['questions']
        })
        
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Failed to parse quiz data: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to generate quiz: {str(e)}'}), 500

# Update the get_quizzes route to include questions
@app.route('/get-quizzes', methods=['GET', 'OPTIONS'])
def get_quizzes():
    if request.method == 'OPTIONS':
        return '', 204
        
    user_id = request.headers.get('Authorization')
    if not user_id:
        return jsonify({'error': 'Authorization required'}), 401
    
    quizzes = Quiz.query.filter_by(user_id=user_id).all()
    return jsonify([{
        'id': quiz.id,
        'title': quiz.title,
        'score': quiz.score,
        'questions': quiz.questions,  # Include questions in response
        'created_at': quiz.created_at.isoformat(),
        'difficulty': quiz.difficulty
    } for quiz in quizzes])

@app.route('/get-quiz/<int:quiz_id>', methods=['GET', 'OPTIONS'])
def get_quiz(quiz_id):
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Get user_id from Authorization header
        user_id = request.headers.get('Authorization')
        if not user_id:
            return jsonify({'error': 'Authorization required'}), 401
        
        # Get quiz and verify ownership
        quiz = Quiz.query.get_or_404(quiz_id)
        if str(quiz.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized access'}), 403
        
        # Return quiz data
        return jsonify({
            'id': quiz.id,
            'title': quiz.title,
            'questions': quiz.questions,
            'score': quiz.score,
            'difficulty': quiz.difficulty,
            'created_at': quiz.created_at.isoformat(),
            'document_id': quiz.document_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch quiz: {str(e)}'}), 500

@app.route('/delete-quiz/<int:quiz_id>', methods=['DELETE', 'OPTIONS'])
def delete_quiz(quiz_id):
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        user_id = request.headers.get('Authorization')
        if not user_id:
            return jsonify({'error': 'Authorization required'}), 401
        
        quiz = Quiz.query.get_or_404(quiz_id)
        if str(quiz.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized access'}), 403
            
        db.session.delete(quiz)
        db.session.commit()
        
        return jsonify({'message': 'Quiz deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete quiz: {str(e)}'}), 500

# Add a route to submit quiz answers
@app.route('/submit-quiz', methods=['POST', 'OPTIONS'])
def submit_quiz():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        quiz_id = data.get('quiz_id')
        answers = data.get('answers', {})
        
        quiz = Quiz.query.get_or_404(quiz_id)
        questions = quiz.questions
        
        # Calculate score
        correct_answers = 0
        total_questions = len(questions)
        feedback = []
        
        for i, question in enumerate(questions):
            user_answer = answers.get(str(i))
            is_correct = user_answer == question['correct_answer']
            correct_answers += 1 if is_correct else 0
            
            feedback.append({
                'question_number': i + 1,
                'correct': is_correct,
                'explanation': question['explanation']
            })
        
        # Update quiz score
        quiz.score = (correct_answers / total_questions) * 100
        db.session.commit()
        
        return jsonify({
            'quiz_id': quiz_id,
            'score': quiz.score,
            'feedback': feedback
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to submit quiz: {str(e)}'}), 500

@app.route('/test-quiz-generation', methods=['POST'])
def test_quiz_generation():
    """Test endpoint with better response handling."""
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        
        document = Document.query.get_or_404(document_id)
        chunks = json.loads(document.chunk_data)
        content_summary = ' '.join(chunks[0].split('\n')[:10])
        
        prompt = f"""Create a multiple choice quiz about this text:

{content_summary}

Create exactly 3 questions following these rules:
1. Each question must be about the text
2. Each question must have exactly 4 options labeled A) through D)
3. Include the correct answer and a brief explanation

Respond with ONLY this JSON format:
{{
    "title": "Quiz Title",
    "questions": [
        {{
            "text": "Question here?",
            "options": [
                "A) Option 1",
                "B) Option 2",
                "C) Option 3",
                "D) Option 4"
            ],
            "correct_answer": "A) Option 1",
            "explanation": "Brief explanation here"
        }}
    ]
}}"""

        response = study_ai.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a quiz generator that outputs only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=4000,
            top_p=0.95
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        return jsonify({
            'raw_response': raw_response,
            'response_length': len(raw_response),
            'document_content': content_summary
        })
        
    except Exception as e:
        logging.error(f"Test endpoint error: {str(e)}")
        return jsonify({
            'error': str(e),
            # 'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    app.run(debug=True)