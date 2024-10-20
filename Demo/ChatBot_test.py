import requests
import json

ARLIAI_API_KEY = "8781f30f-6573-4b94-9cc8-6875a5dd5433"

url = "https://api.arliai.com/v1/completions"
system_prompt = """
# Character
You're an expert in web designing with a focus on HTML, CSS, and Tailwind CSS. You modify website elements based on user requests.

## Skills
### Skill 1: Modify Website Contents
- Make color, text, and style changes to specified elements.
- Ensure the modifications adhere to Tailwind CSS conventions.

### Skill 2: Add New Elements
- Integrate new HTML elements and style them using Tailwind CSS.

### Skill 3: Debug and Optimize Code
- Identify inefficiencies or issues in the code and provide optimized solutions using your expertise in HTML, CSS, and Tailwind CSS.

## Constraints
- Only use HTML, CSS, and Tailwind CSS for modifications.
- Respond only to modification-related requests.
- Provide both a confirmation response and the updated code in your output.

# Example Input
{
"request": "change the heading world of AI to red",
"code":"<body><div>
<h1 class="text-black">world of AI</h1>
</div></body>"
}

# Example Output
{
"response": "Sure, I have changed it.",
"code": "<body><div>
<h1 class="text-red">world of AI</h1>
</div></body>"
}
"""

prompt = """
{
  "request": "Center the text vertically, change the Brand to Jesus High Fashion, modify the lorem ipsum text section content to introduction about Jesus High Fashion",
  "code": "<body class="flex flex-col items-center justify-center bg-red-500 w-full h-screen">
<div class="w-full h-screen relative" style="height:100vh">
  <img src="https://images.pexels.com/photos/28219391/pexels-photo-28219391/free-photo-of-the-dolomites-in-italy.jpeg" alt="" class="object-cover object-top absolute top-0 left-0 w-full h-full z-10">
  <div class="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-transparent via-transparent to-black z-20 p-5 text-white text-center">
    <h1 class="text-6xl font-extrabold mb-5">Jesus High Fashion</h1>
    <p class="text-lg leading-relaxed mt-10 mb-10">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut eleifend scelerisque nisi quis aliquam. Sed at tempor ex. Maecenas at 
aliquet augue. Etiam nisl odio, commodo vel nibh sed, gravida eleifend magna.</p>
    <p class="text-lg leading-relaxed mt-10 mb-10">Introduction about Jesus High Fashion</p>
    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Get Started</button>
  </div>
</div>
</body>"
}
"""

payload = json.dumps({
    "model": "Meta-Llama-3.1-8B-Instruct",
    "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "repetition_penalty": 1.1,
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 1024,
    "stream": False
})
headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {ARLIAI_API_KEY}"
}

response = requests.request("POST", url, headers=headers, data=payload)
obj = response.json()
text = obj['choices'][0]['text']

# parse = response.text.split('data: ')
# parse = parse[1:len(parse) - 1]
# text = ''

# for i in range(0, len(parse)):
#     parse[i] = json.loads(parse[i])
#     text += parse[i]['choices'][0]['text']

# # text = json.loads(text)

print(text)
