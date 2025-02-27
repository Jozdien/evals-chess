from openai import OpenAI
client = OpenAI()

def finetune_create(file, model, batch_size='auto', lr_multiplier='auto', n_epochs='auto'):
  finetune_obj = client.fine_tuning.jobs.create(
    training_file=file,
    model=model,
    method={
      "type": "supervised",
      "supervised": {
        "hyperparameters": {
          "batch_size": batch_size,
          "learning_rate_multiplier": lr_multiplier,
          "n_epochs": n_epochs
        },
      }
    }
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
  file = "file-EVD177kQsPt8xVuixKwNFe"
  lr_multipliers = [3]
  n_epochs = [5]
  exclude = [(0.01, 1), (0.01, 3), (0.01, 5), (0.01, 10), (0.2, 1), (0.2, 3), (0.2, 5), (0.2, 10)]

  for lr, ep in [(l, e) for l in lr_multipliers for e in n_epochs if (l, e) not in exclude]:
    finetune_create(file=file, model=model, lr_multiplier=lr, n_epochs=ep)
  # list_finetunes()