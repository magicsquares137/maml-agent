from dataclasses import dataclass 

@dataclass 
class Template:
    """
    This prompt template is taken from paper https://arxiv.org/abs/2407.18901 appendix
    """
    template: str = """USER:
    I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

    To do this, you will need to interact with app/s (e.g., spotify, venmo etc) using their associated APIs on my behalf. 
    For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the 
    python code and the environment will execute it and show you the result, based on which, you will write python code 
    for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

    Here are three key APIs that you need to know to get more information

    # To get a list of apps that are available to you.
    print(apis.api_docs.show_app_descriptions())

    # To get the list of apis under any app listed above, e.g. spotify
    print(apis.api_docs.show_api_descriptions(app_name='spotify'))

    # To get the specification of a particular api, e.g. spotify app's login api
    print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))

    Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now 
    generate code, that I will execute, to solve the task. Let's start with the task

    My name is: Jane Doe. My personal email is jane@example.com and phone number is 555-1234.

    Task: How many playlists do I have in Spotify?

    ASSISTANT:
    Okay. Lets first find which APIs are available to use in Spotify.
    Code:
    ```python
    print(apis.api_docs.show_api_descriptions(app_name='spotify'))
    ```

    USER:
    Output:
    ```
    [
    ...
    "login : Login to your account.",
    "logout : Logout from your account.",
    ...
    ]
    ```

    ASSISTANT:
    Okay. Looks like I can use the `login` api. Lets find its specifications.
    Code:
    ```python
    print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
    ```

    USER:
    Output:
    ```
    {{
    "app_name": "spotify",
    "api_name": "login",
    "parameters": [...],
    "response_schema": {{"access_token": "string"}}
    }}
    ```

    ASSISTANT:
    Okay, I'll need the password. Let me get that from the supervisor.
    Code:
    ```python
    passwords = apis.supervisor.show_account_passwords()
    spotify_password = None
    for account in passwords:
        if account["account_name"] == "spotify":
            spotify_password = account["password"]
            break

    response = apis.spotify.login(username="jane@example.com", password=spotify_password)
    spotify_access_token = response["access_token"]
    print("Logged in successfully.")
    ```

    USER:
    Output:
    ```
    Logged in successfully.
    ```

    ASSISTANT:
    The `show_playlist_library` API seems to be the one we need. Let's check its specifications.
    Code:
    ```python
    print(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_playlist_library'))
    ```

    USER:
    Output:
    ```
    {{
    "app_name": "spotify",
    "api_name": "show_playlist_library",
    "parameters": [
      {{"name": "access_token", "type": "string", "required": true}},
      {{"name": "page_index", "type": "integer", "required": false, "default": 0}}
    ],
    "response_schema": [
      {{"playlist_id": 1, "title": "string", "song_ids": [1]}}
    ]
    }}
    ```

    ASSISTANT:
    I need to page through all the playlists to get the list of playlists and save it in `playlists`.
    Code:
    ```python
    page_index = 0
    playlists = []
    while page_index < 10:
        playlist_page = apis.spotify.show_playlist_library(access_token=spotify_access_token, page_index=page_index)
        if playlist_page:
            playlists.extend(playlist_page)
            page_index += 1
        else:
            break
    num_playlists = len(playlists)
    print(num_playlists)
    ```

    USER:
    Output:
    ```
    23
    ```

    ASSISTANT:
    Now that the task is completed, I can call apis.supervisor.complete_task(). Since this task has an answer to be returned, I will pass that as an argument.
    Code:
    ```python
    apis.supervisor.complete_task(answer=num_playlists)
    ```

    USER:
    Output:
    Marked the active task complete.

    ----------------------------------------------

    USER:
    **Key instructions**:
    (1) Make sure to end code blocks with ``` followed by a newline.
    (2) Remember you can use the variables in your code in subsequent code blocks.
    (3) Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example above are not valid anymore.
    (4) You can use the "supervisor" app to get information about my accounts and use the "phone" app to get information about friends and family.
    (5) Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.
    (6) Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
    (7) Many APIs return items in "pages". Make sure to run through all the pages by looping over `page_index`.
    (8) Once you have completed the task, make sure to call apis.supervisor.complete_task(). If the task asked for some information, 
        return it as the answer argument, i.e. call apis.supervisor.complete_task(answer=<answer>). Many tasks do not require an answer, 
        so in those cases, just call apis.supervisor.complete_task() i.e. do not pass any argument.

    USER:
    Using these APIs, now generate code to solve the actual task:

    My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.

    Task: {task_instruction}"""

    def format_prompt(self, first_name, last_name, email, phone_number, task_instruction):
        return self.template.format(            
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone_number=phone_number,
            task_instruction=task_instruction
        )