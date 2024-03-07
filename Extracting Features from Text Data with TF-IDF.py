from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer from scikit-learn for text feature extraction.

class TfidfDocumentAnalyzer:
    def __init__(self, documents):
        self.documents = documents  # Store the input list of documents.
        self.vectorizer = TfidfVectorizer()  # Initialize the TF-IDF vectorizer.
        self.features = None  # Placeholder for the document feature matrix.
        self.feature_names = None  # Placeholder for the feature names.

    def fit_transform_documents(self):
        # Convert documents into a matrix of TF-IDF features.
        self.features = self.vectorizer.fit_transform(self.documents).toarray()
        # Extract and store feature names.
        self.feature_names = self.vectorizer.get_feature_names_out()

    def print_tfidf_values(self):
        # Iterate over documents and their TF-IDF vectors to print values.
        for i, document in enumerate(self.documents):
            print("Document:", document)  # Print the current document.
            for j, feature in enumerate(self.feature_names):
                tfidf_value = self.features[i][j]  # Extract TF-IDF value for the feature.
                if tfidf_value > 0:
                    # Print feature and its TF-IDF value if it's greater than zero.
                    print(" Feature:", feature, " TF-IDF:", tfidf_value)

def main():
    # List of documents to be analyzed.
    documents = [
        "I love eating pizza",
        "I enjoy playing soccer",
        "Soccer is a popular sport",
        "Pizza and soccer are my favorite things"
    ]

    # Initialize the document analyzer with the list of documents.
    document_analyzer = TfidfDocumentAnalyzer(documents)
    document_analyzer.fit_transform_documents()  # Convert documents to TF-IDF features.
    document_analyzer.print_tfidf_values()  # Print the TF-IDF values for each document.

if __name__ == "__main__":
    main()  # Run the main function if the script is executed.
