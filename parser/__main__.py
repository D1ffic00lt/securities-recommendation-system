def main():
    from utils.api import APIParser

    with open("../secrets/tinkoff_token.txt", "r") as f:
        api_token = f.read().strip()

    parser = APIParser(api_token)
    shares = parser.parse_shares()


if __name__ == "__main__":
    main()
