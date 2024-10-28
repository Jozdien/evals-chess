from openai import OpenAI
client = OpenAI()

def upload_file(filename, purpose="fine-tune"):
  file_obj = client.files.create(
    file=open(filename, "rb"),
    purpose=purpose
  )
  print(file_obj)

def list_files():
  files = client.files.list()
  for file in files:
    print(file)

if __name__ == '__main__':
  upload_file(filename='chess_db_elo_1700_1900_truncated.jsonl', purpose='fine-tune')
  # list_files()