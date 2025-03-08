import requests
import json
import os

def test_study_ai():
    base_url = 'http://127.0.0.1:5000'
    
    # 1. Register user
    register_response = requests.post(
        f'{base_url}/register',
        json={
            'username': 'testuser',
            'password': 'password123'
        }
    )
    print("Registration:", register_response.json())
    
    if 'error' in register_response.json():
        raise Exception("Registration failed")
        
    user_id = register_response.json()['user_id']
    
    # 2. Upload document
    test_pdf = "test.pdf"  # Make sure this file exists
    if not os.path.exists(test_pdf):
        raise Exception(f"Test file {test_pdf} not found")
        
    with open(test_pdf, 'rb') as f:
        upload_response = requests.post(
            f'{base_url}/upload-document',
            files={'file': f},
            data={'user_id': user_id}
        )
    print("Document upload:", upload_response.json())
    
    if 'error' in upload_response.json():
        raise Exception("Document upload failed")
        
    document_id = upload_response.json()['document_id']
    
    # 3. Test Quiz Generation Endpoint
    # Test the raw response first
    test_response = requests.post(
        f'{base_url}/test-quiz-generation',
        json={'document_id': document_id}
    )
    print("Test response:", json.dumps(test_response.json(), indent=2))

    # If the raw response looks good, try the quiz generation
    if 'error' not in test_response.json():
        quiz_response = requests.post(
            f'{base_url}/generate-quiz',
            json={
                'document_id': document_id,
                'num_questions': 3,
                'difficulty': 'medium'
            }
        )
        print("Quiz generation:", json.dumps(quiz_response.json(), indent=2))
    
    if 'error' in quiz_response.json():
        raise Exception("Quiz generation failed")
        
    quiz_id = quiz_response.json()['quiz_id']
    
    # 4. Submit quiz answers
    answers = {
        '0': 'A',
        '1': 'B',
        '2': 'True'
    }
    
    submit_response = requests.post(
        f'{base_url}/submit-quiz',
        json={
            'quiz_id': quiz_id,
            'answers': answers
        }
    )
    print("Quiz submission:", json.dumps(submit_response.json(), indent=2))

if __name__ == '__main__':
    try:
        test_study_ai()
    except Exception as e:
        print(f"Test failed: {str(e)}")