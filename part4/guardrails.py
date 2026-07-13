import re

def has_pii(text):
    """
    Checks if the input text contains Personal Identifiable Information (PII)
    like email addresses or phone numbers.
    """
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    
    return bool(re.search(email_pattern, str(text)) or re.search(phone_pattern, str(text)))