import os

if __name__ == "__main__":
    # sanity check 
    os.system('curl http://54.163.206.95:80/')
    print("\n\n")

    # making a post request
    os.system('curl -X POST http://54.163.206.95:500/prediction -H "Content-Type: application/json" -d @inference/payload.json')
    print("\n\n")