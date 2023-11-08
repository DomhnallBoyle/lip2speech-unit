import smtplib
from email.mime.text import MIMEText

import config


def send_email(sender, receivers, subject, content):
    assert type(receivers) is list

    for param in ['EMAIL_HOST', 'EMAIL_PORT', 'EMAIL_USERNAME', 'EMAIL_PASSWORD']:
        v = getattr(config, param)
        if not v:
            raise Exception(f'Failed to send email: config.{param} is not set...')

    message = MIMEText(content)
    message['Subject'] = subject
    message['From'] = sender

    server = smtplib.SMTP(config.EMAIL_HOST, port=config.EMAIL_PORT)
    server.starttls()
    server.login(config.EMAIL_USERNAME, config.EMAIL_PASSWORD)
    server.sendmail(sender, receivers, message.as_string())
    server.quit()
