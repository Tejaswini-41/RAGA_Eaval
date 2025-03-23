# from github import Github
# import chromadb
# import openai
# from chromadb.utils import embedding_functions
# import httpx
# import uuid


# # Create embeddings with the names of files changes for respective PRs. Get the names of files changes in current pull request. Query DB with the same to get the matching PRs.
# # Use the contents of those PRs in prompt to ask for review on current PR.

# #Replace 'your_access_token' with your GitHub personal access token
# github_access_token = 'your_access_token'
# g = Github(f"{github_access_token}")

# #Replace 'repo_owner' and 'repo_name' with the owner and name of the repository
# repo_owner = 'torvalds'
# repo_name = 'linux'
# repo = g.get_repo(f"{repo_owner}/{repo_name}")
# Changed_Files = []
# Pull_Requests = []
# count = 0
# #Iterate through all open pull requests in the repository
# #repo.get_pull()
# for pull_request in repo.get_pulls(state='closed'):
#     print(f"Checking Pull Request #{pull_request.number} - {pull_request.title}")

#     #Get the list of files changed in the pull request
#     files = pull_request.get_files()
#     sFiles = ""
#    #Iterate through each file and print the added lines
#     for file in files:
#         print(f"File: {file.filename}")
#         sFiles = sFiles + "," + file.filename
#         print("\n" + "=" * 50 + "\n")  # Separating each file's output
#     Changed_Files += [sFiles]
#     Pull_Requests += [pull_request.number]
#     count += 1
#     print("\n" + "=" * 50 + "\n")  # Separating each pull request's output
#     if count > 10:
#         break
# print(Changed_Files)
# print(Pull_Requests)

# httpx_client = httpx.Client(http2=True, verify=False)

# CHROMA_DATA_PATH = "chroma_data/"
# EMBED_MODEL = "all-mpnet-base-v2"
# #EMBED_MODEL = "text-embedding-ada-002"
# COLLECTION_NAME = str(uuid.uuid4())

# client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name=EMBED_MODEL
# )

# collection = client.create_collection(
#     name=COLLECTION_NAME,
#     embedding_function=embedding_func,
#     metadata={"hnsw:space": "cosine"},
# )

# collection.add(
#     documents=Changed_Files,
#     ids=[f"id{i}" for i in range(len(Changed_Files))],
#     metadatas=[{"genre": g} for g in Pull_Requests]
# )

# client2 = openai.AzureOpenAI(
#         api_version="2023-07-01-preview",
#         azure_endpoint="{Azure_Endpoint}",
#         api_key= "{OpenAI_API_Key}",
#         http_client=httpx_client
# )

# pull_request1 = repo.get_pull(896)
# files1 = pull_request1.get_files()
# filenames = ""
# for file in files1:
#     #print(f"File: {file.filename}")
#     filenames += file.filename + ","

# def text_embedding(text):
#     response = client2.embeddings.create(model="text-embedding-ada-002", input=text)
#     return response.data[0].embedding[:768]

# vector=text_embedding(filenames)

# query_results = collection.query(
#     query_embeddings=vector,
#     n_results=1
# )


# print(query_results.keys())

# print(query_results.values())

# print(query_results["documents"])

# print(query_results["ids"])

# print(query_results["distances"])

# print(query_results["metadatas"])

# metadata = query_results["metadatas"]
# List_PR_number = metadata.pop(0)
# PR_numbers = List_PR_number[0]
# PR_number = PR_numbers.get("genre")

# pull_request2 = repo.get_pull(PR_number)
# files = pull_request2.get_files()

# def get_File_Changes(files):
#     Final_PR = ""
#     for file in files:
#         patch_content = file.patch
#             #Split the patch content into lines
#         if patch_content is not None:
#             patch_lines = patch_content.split('\n')
            
#             patch_lines_before = patch_lines
#             patch_lines_after = patch_lines

#             added_lines = [line for line in patch_lines if line.startswith('+') and not line.startswith('+++')]
#             removed_lines = [line for line in patch_lines if line.startswith('-') and not line.startswith('---')]

#             #Identify and print added lines
#             added_lines_print = [line[1:] for line in patch_lines if line.startswith('+') and not line.startswith('+++')]
#             #add similar function for removed lines
#             removed_lines_print = [line[1:] for line in patch_lines if line.startswith('-') and not line.startswith('---')]

#             print("Added Lines:")
#             for added_line in added_lines_print:
#                 print(added_line)

#             print("Removed Lines:")
#             for removed_line in removed_lines_print:
#                 print(removed_line)

#             for added_line in added_lines:
#                 if patch_lines_before is not None:
#                     patch_lines_before = patch_lines_before.remove(added_line)

#             for removed_line in removed_lines:
#                 if patch_lines_after is not None:
#                     patch_lines_after = patch_lines_after.remove(removed_line)

#             print("Before:")
#             print(patch_lines_before)
#             Before_PR = ""
#             if patch_lines_before is not None:
#                 #Before_PR = Before_PR.join([str[item] for item in patch_lines_before])
#                 Before_PR = Before_PR.join(patch_lines_before)
#             print("After:")
#             print(patch_lines_after)
#             After_PR = ""
#             if patch_lines_after is not None:
#                 After_PR = After_PR.join(patch_lines_after)
#             if len(Before_PR) > 0 or len(After_PR) > 0:
#                 Final_PR = Final_PR + "Before Pull Request: " + Before_PR + " After Pull Request: " + After_PR + "\n"            
#     return Final_PR

# Final_PR2 = get_File_Changes(files)
# Final_PR1 = get_File_Changes(files1)

# res = "\n".join(str(item) for item in query_results['documents'][0])
# #prompt=f'```{res}```Based on the data in ```, Please suggest the changes in After pull request text for following. Before pull request text- }}	break; case USB_ID(0x046d, 0x0807): 	case USB_ID(0x046d, 0x0808):	case USB_ID(0x046d, 0x0809): After pull request text- }}	break; case USB_ID(0x05ac, 0x110a): 		if(!strcmp(kctl->id.name, "PCM Playback Volume")){{			/* Set PCM Playback Volume for channel 1 and 2 to maximum supported volume */			snd_usb_set_cur_mix_value(cval, 1, 0, cval->max);			snd_usb_set_cur_mix_value(cval, 2, 1, cval->max);		}}		break;case USB_ID(0x046d, 0x0807): 	case USB_ID(0x046d, 0x0808):	case USB_ID(0x046d, 0x0809):'
# prompt=f'Based on the content in ' + Final_PR2 + ', please suggest the changes in ' + Final_PR1

# messages = [
#         {"role": "system", "content": "You answer questions about Pull Requests."},
#         {"role": "user", "content": prompt}
# ]

# response = client2.chat.completions.create(
#         model="gpt-35-turbo",
#         messages=messages,
#         temperature=0
# )

# response_message = response.choices[0].message.content

# print(response_message)