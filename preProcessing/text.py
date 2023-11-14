import re
import unicodedata
import email

def extract_mail(text):
    msg = email.message_from_string(text)
    message_body = ""

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

    return message_body

# Nome do arquivo de texto com os emails
filename = "raw.txt"

# Lista para armazenar os textos das mensagens
messages = []

# Lê o arquivo de texto linha por linha
with open(filename, "r", encoding="utf-8") as file:
    remove_header = False
    email_text = ""
    for line in file:
        
        if not line:
            # Skip empty lines
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
        
        if line.startswith(">" or "[IMAGE]"):
            # Skip lines starting with ">"
            continue
        if line.startswith("Message-ID:"):
            # Encontrou o início de um novo email, extrai o texto e adiciona à lista
            if email_text:
                message = extract_mail(email_text)
                messages.append(message)
            email_text = line
        else:
            # Concatena as linhas do email
            email_text += line


# Extrai o último email
if email_text:
    message = extract_mail(email_text)
    messages.append(message)

with open("new_data", "w+") as file:
    for message in messages:
        # write each item on a new line
        file.write("%s\n" % message)

with open("new_data", 'r') as file:
    lines = file.readlines()

    # Utilizar expressões regulares para filtrar as linhas
    prefix_pattern  = re.compile(r"^\s*(To:|cc:|Subject:|From:)", re.IGNORECASE)
    space_pattern = re.compile(r"^\s+")

    filtered_lines = [line for line in lines if not prefix_pattern.match(line)]
    filtered_lines = [space_pattern.sub('', line) for line in filtered_lines]


    # Escrever as linhas filtradas em um novo arquivo
    with open("new_data", 'w') as file:
        file.writelines(filtered_lines)
