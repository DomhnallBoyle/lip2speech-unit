common_name=${COMMON_NAME:-127.0.0.1}

openssl req -subj '/CN='$common_name -addext "subjectAltName = IP:$common_name" -x509 -nodes -days 3650 -newkey rsa:2048 -keyout key.pem -out cert.pem
