
if __name__ == "__main__":
    try:
        import QR1
        print("QR1 found ! Environment is ready.")
    except ImportError:
        print("QR1 not found. Please check your environment setup.")
