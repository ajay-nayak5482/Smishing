if __name__ == "__main__":
    import os
    import sys
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import data_preprocessing


    # Ensure the script is run with Python 3
    if sys.version_info[0] < 3:
        print("This script requires Python 3.")
        sys.exit(1)

    # Set the working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    # Import and run the main functionality of the application
    # from app import main_functionality
    # main_functionality.run()
    # Download NLTK data if not already present
    try:
        stopwords.words('english')
        word_tokenize("test")
    except LookupError:
        print("Downloading NLTK data (stopwords, punkt)...")        
        nltk.download('stopwords')
        nltk.download('punkt')
        print("NLTK data downloaded.")
    data_preprocessing.preprocess_datasets(script_dir)
    


