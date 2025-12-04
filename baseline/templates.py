from dataclasses import dataclass 

# @dataclass 
# class Template:
#     template: str = """USER:
#     I am your supervisor and you are a super intelligent AI Assistant whose job is to
#     achieve my day-to-day tasks completely autonomously.

#     To do this, you will need to interact with app/s (e.g., spotify, venmo etc) using
#     their associated APIs on my behalf. For this you will undertake a *multi-step
#     conversation* using a python REPL environment.

#     Here are three key APIs that you need to know to get more information:
#     ```python
#     print(apis.api_docs.show_app_descriptions())
#     print(apis.api_docs.show_api_descriptions(app_name='spotify'))
#     print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
#     ```

#     Each code execution will produce an output that you can use in subsequent calls.

#     Using these APIs, you can now generate code, that I will execute, to solve the task.

#     My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.

#     Task: How many playlists do I have in Spotify?

#     A:
#     Okay. Lets first find which APIs are available to use in Spotify.
#     Code:
#     ```python
#     print(apis.api_docs.show_api_descriptions(app_name='spotify'))
#     ```

#     USER:
#     Output:
#     ```
#     [
#     ...
#     "login : Login to your account.",
#     ...
#     ]
#     ```

#     A:
#     Okay. Looks like I can use the `login` api. Lets find its specifications.
#     Code:
#     ```python
#     print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
#     ```

#     USER:
#     Output:
#     ```
#     {{
#       "app_name": "spotify",
#       "api_name": "login",
#       "parameters": [...],
#       "response_schema": {{"access_token": "string"}}
#     }}
#     ```

#     A:
#     Okay, I'll need the password. Let me get that from the supervisor.
#     Code:
#     ```python
#     passwords = apis.supervisor.show_passwords()
#     spotify_password = None
#     for app_name, pwd in passwords:
#         if app_name == "spotify":
#             spotify_password = pwd
#             break

#     response = apis.spotify.login(username="{email}", password=spotify_password)
#     spotify_access_token = response["access_token"]
#     print(f"Logged in successfully.")
#     ```

#     USER:
#     Output:
#     ```
#     Logged in successfully.
#     ```

#     A:
#     The `show_playlist_library` API seems to be the one we need.
#     Code:
#     ```python
#     page_index = 0
#     playlists = []
#     while page_index < 10:
#         playlist_page = apis.spotify.show_playlist_library(
#             access_token=spotify_access_token, 
#             page_index=page_index
#         )
#         if playlist_page:
#             playlists.extend(playlist_page)
#             page_index += 1
#         else:
#             break
#     num_playlists = len(playlists)
#     print(num_playlists)
#     ```

#     USER:
#     Output:
#     ```
#     23
#     ```

#     A:
#     Now that the task is completed, I can call apis.supervisor.complete_task().
#     Code:
#     ```python
#     apis.supervisor.complete_task(answer=num_playlists)
#     ```

#     USER:
#     Output:
#     Marked the active task complete.

#     -----------------------------------------------------------

#     USER:
#     **Key instructions**:
#     (1) End code blocks with ``` followed by a newline.
#     (2) You can reuse variables from previous code blocks.
#     (3) Email addresses and tokens in the example above are not valid anymore.
#     (4) Use "supervisor" app for account info and "phone" app for contacts.
#     (5) Always check API specs with apis.api_docs.show_api_doc before calling.
#     (6) Write small chunks of code, one per step.
#     (7) Many APIs return "pages". Loop through all pages with `page_index`.
#     (8) Call apis.supervisor.complete_task() when done. Pass answer= if needed.

#     USER:
#     Using these APIs, now generate code to solve the actual task:

#     My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.

#     Task: {task_instruction}"""

#     def format_prompt(self, first_name, last_name, email, phone_number, task_instruction):
#         return self.template.format(            
#             first_name=first_name,
#             last_name=last_name,
#             email=email,
#             phone_number=phone_number,
#             task_instruction=task_instruction
#             )
# templates.py
@dataclass 
class Template:
    template: str = """USER:
    I am your supervisor and you are a super intelligent AI Assistant whose job is to
    achieve my day-to-day tasks completely autonomously.

    To do this, you will need to interact with app/s (e.g., spotify, venmo etc) using
    their associated APIs on my behalf. For this you will undertake a *multi-step
    conversation* using a python REPL environment.

    Here are three key APIs that you need to know to get more information:
    ```python
    print(apis.api_docs.show_app_descriptions())
    print(apis.api_docs.show_api_descriptions(app_name='spotify'))
    print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
    ```

    Each code execution will produce an output that you can use in subsequent calls.

    Using these APIs, you can now generate code, that I will execute, to solve the task.

    My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.

    Task: How many playlists do I have in Spotify?

    A:
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
    ...
    ]
    ```

    A:
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

    A:
    Okay, I'll need the password. Let me get that from the supervisor.
    Code:
    ```python
    passwords = apis.supervisor.show_account_passwords()
    spotify_password = None
    for account in passwords:
        if account["account_name"] == "spotify":
            spotify_password = account["password"]
            break

    response = apis.spotify.login(username="{email}", password=spotify_password)
    spotify_access_token = response["access_token"]
    print(f"Logged in successfully.")
    ```

    USER:
    Output:
    ```
    Logged in successfully.
    ```

    A:
    The `show_playlist_library` API seems to be the one we need.
    Code:
    ```python
    page_index = 0
    playlists = []
    while page_index < 10:
        playlist_page = apis.spotify.show_playlist_library(
            access_token=spotify_access_token, 
            page_index=page_index
        )
        if playlist_page:
            playlists.extend(playlist_page)
            page_index += 1
        else:
            break
    num_playlists = len(playlists)
    print(f"Total playlists: {{num_playlists}}")
    ```

    USER:
    Output:
    ```
    Total playlists: 23
    ```

    A:
    Now that I have the answer, I'll complete the task with the answer.
    Code:
    ```python
    apis.supervisor.complete_task(answer=num_playlists)
    ``` 

    USER:
    Output:
    Marked the active task complete.

    -----------------------------------------------------------

    USER:
    **Key instructions**:
    (1) End code blocks with ``` followed by a newline.
    (2) You can reuse variables from previous code blocks.
    (3) Email addresses and tokens in the example above are not valid anymore.
    (4) Use "supervisor" app for account info and "phone" app for contacts.
    (5) Always check API specs with apis.api_docs.show_api_doc before calling.
    (6) Write small chunks of code, one per step.
    (7) Many APIs return "pages". Loop through all pages with `page_index`.
    (8) CRITICAL: When the task asks for information, you MUST pass it as answer=<value> to complete_task(). If the task asks you to DO something (not return information), call complete_task() with NO answer parameter.
    (9) CRITICAL: Before calling complete_task(), verify that you have actually made the required changes. Do not complete the task prematurely.

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