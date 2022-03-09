"""Interface for reading and writing the secret words file in javascript format"""

SECRET_WORDS_FILE = "static/assets/js/secretWords.js"


def read():
    """
    Reads the secret words file.
    Format: JavaScript file, UTF-8 encoded, leading and trailing whitespace is ignored.
    This method does not support JavaScript comments.
    """
    with open(SECRET_WORDS_FILE, encoding="utf-8") as secret_words:
        return [l.strip().strip('",') for l in secret_words if '"' in l]


if __name__ == "__main__":
    print(read())
