import re
import email
import json

json_to_tuning = []

def save_emails_to_txt(filename):
    with open(filename, "w") as file:
        json.dump(json_to_tuning, file,  indent=4)

def count_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        character_count = len(text)
        return character_count

DATA_AMOUNT = 13

## https://www.corpusdata.org/
## https://skylion007.github.io/OpenWebTextCorpus/
filename = "base_mail.txt"
general_base_path = f'./general/train.%s.en'
final_file = "data.txt"
mail_fine_tuning = "tuning_data.txt"

general_char_count = 0
mail_char_count = 0

for i in range(1, DATA_AMOUNT):
    general_char_count += count_characters(general_base_path % i)
mail_char_count = count_characters(filename)

print(f"OpenWebTextCorpus = {general_char_count//1e6}M characteres.\nRandomMailChar = {mail_char_count//1e6}M characteres.\nTotal = {general_char_count//1e6 + mail_char_count//1e6}M chars")


# Lista para armazenar os textos das mensagens
messages = []

def extract_mail(text):
    msg = email.message_from_string(text)
    message_body = ""

    subject = msg["Subject"]
    if msg.is_multipart():
        # Traverse all parts of the email
        for part in msg.walk():
            # Check if the part is of type text/plain
            if part.get_content_type() == "text/plain":
                # Decode the payload and append it to the message body
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset()
                if charset:
                    message_body += payload.decode(charset)
                else:
                    message_body += payload.decode()
    else:
        # If the email is not multipart, directly obtain the body
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset()
        if charset:
            message_body = payload.decode(charset)
        else:
            message_body = payload.decode()

     # Cria um dicionário com o subject e o body do email
    if subject != None:
        if subject != "" and subject != "Re:":
            email_dict = {
                "subject": subject.replace('Re:', '').strip(),
                "body": message_body.replace('[IMAGE]', '')
            }

            # Adiciona o dicionário à variável global emails_json
            json_to_tuning.append(email_dict)

    return message_body.replace('[IMAGE]', '')



# Lê o arquivo de texto linha por linha
with open(filename, "r", encoding="utf-8") as file:
    remove_header = False
    email_text = ""
    for line in file:
        
        if not line:
            # Skip empty lines
            continue
        
                
        if line.startswith(">" or "[IMAGE]"):
            # Skip lines starting with ">"
            continue

        if line.startswith("Subject:"):
            before_hader = remove_header
            remove_header = False
            
            if before_hader: continue
        if remove_header:
            # Skip the lines to remove the original message header
            continue

        if line.startswith("-----Original Message-----") or line.startswith(" -----"):
            # Start removing the original message header
            remove_header = True
            continue

        elif line.startswith("From:"):
            # Stop removing the header when a new "From:" line is encountered
            remove_header = False

        if line.startswith("Message-ID:"):
            # Encontrou o início de um novo email, extrai o texto e adiciona à lista
            if email_text:
                message = extract_mail(email_text)
                messages.append(message)
            email_text = line
        else:
            # Concatena as linhas do email
            email_text += line


save_emails_to_txt("saved_emails_to_tuning.txt")

# Find the last email data
if email_text:
    message = extract_mail(email_text)
    messages.append(message)

with open(final_file, "w+") as file:
    for message in messages:
        # write each item on a new line
        file.write("%s\n" % message)

with open(final_file, 'r') as file:
    lines = file.readlines()

    # Filter for unwanted string
    prefix_pattern  = re.compile(r"^\s*(To:|cc:|Subject:|From:)", re.IGNORECASE)
    space_pattern = re.compile(r"^\s+")

    filtered_lines = [line for line in lines if not prefix_pattern.match(line)]
    filtered_lines = [space_pattern.sub('', line) for line in filtered_lines]

    with open(final_file, 'w') as file:
        file.writelines(filtered_lines)

    with open(mail_fine_tuning, 'w') as file:
        file.writelines(filtered_lines)    



for i in range(1, DATA_AMOUNT):
    with open(general_base_path %  i, 'r') as file:
        lines = file.readlines()

        filtered_lines = [line.replace(" @", '') for line in lines]
        prefix_pattern  = re.compile("^@@.*", re.IGNORECASE)

        filtered_lines = [line[14:] for line in lines if prefix_pattern.match(line)]

        with open(final_file, 'a') as file:
            file.writelines(filtered_lines)


final_data_char_count = count_characters(final_file)
print(f"final_data_char_count = {final_data_char_count} ({final_data_char_count//1e6}M) caracteres.")
