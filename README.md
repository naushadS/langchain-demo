# Langchain Demo with Huggingface deployment

Followed youtube tutorial:  
https://www.youtube.com/watch?v=qMIM7dECAkc&pp=ygUPa3Jpc2ggbGFuZ2NoYWlu  

Work laptop may give ssl certificate error due to vpn.
Solution : https://community.openai.com/t/ssl-certificate-verify-failed/32442
1. Download certificate on platform.openapi.com page
2. open in text editor - copy everything -> certificate_contents
3. run python -m certifi - it shows a filepath
4. append certificate_contents to this file
5. create a new key in .env file : "REQUESTS_CA_BUNDLE" with value as filepath
6. load the .env file
7. done 