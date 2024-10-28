from openai import OpenAI
client = OpenAI()

def finetune_create(file, model):
  finetune_obj = client.fine_tuning.jobs.create(
    training_file=file,
    model=model
  )
  print(finetune_obj)

def list_finetunes(limit=10):
  jobs = client.fine_tuning.jobs.list(limit=limit)
  for job in jobs:
    print(job)

def finetune_state(id):
  print(client.fine_tuning.jobs.retrieve(id))

def finetune_cancel(id):
  print(client.fine_tuning.jobs.cancel(id))

def finetune_list_events(id, limit=10):
  events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=id, limit=limit)
  for event in events:
    print(event)

def finetune_delete(model_name):
  print(client.models.delete(model_name))

if __name__ == "__main__":
  model = "gpt-4o-2024-08-06"
  file = "file-4eE8s68QghFyMDkCd25LzrVZ"
  finetune_create(file=file, model=model)