exit()




  # get files
  print(extList)
  _, _, fn = next(os.walk(os.getcwd() + "/footage/input"), (None, None, []))
  fn = list(filter(lambda n: n[-4:] in extList or n[-3:] in extList, fn))
  # worked: print(fn)
  workD = kPath(dirname(abspath(__file__)) + '/footage')
  inD = data['file_management_options']['default_input_folder']
  if '[workD]' in inD:
      inD = inD.replace('[workD]\\', '').replace('[workD]', '')
      inD = workD.append(inD)
  else:
      inD = kPath(inD)
  outD = workD.append('output')
  #for n in fn:
      startSession = datetime.datetime.now()
      session = kss()
      #session.runCode(sessID, os.getcwd() + '/footage/input', os.getcwd(), os.getcwd() + 'footage/output', os.getcwd() + 'footage/input/' + n)
      x.runCode(sessID, inD, workD, outD, [v])
      stopSession = datetime.datetime.now()
      print(f'session time = {stopSession - startSession}')

  ###### old code
  workD = kPath(dirname(abspath(__file__)) + "\\footage")
  inD = data['file_management_options']['default_input_folder']
  if '[workD]' in inD:
      inD = inD.replace('[workD]\\', '').replace('[workD]', '')
      inD = workD.append(inD)
  else:
      inD = kPath(inD)
  outD = workD.append('output')

  print(bordered(f'{program_name} version {program_version}'))
  import datetime
  a = datetime.datetime.now()
  if individual_videos:
      fSet = list(os.listdir(inD.aPath()))
      print(fSet)
      vids = sorted(list(filter(lambda v: v[-4:].lower() in extList, fSet)))
      vids = list(map(lambda v: inD.append(v), vids))
      print(f'bp ({len(vids)})')
      for v in vids:
          x = kss()
          print('next step')
          x.runCode(sessID, inD, workD, outD, [v])
  else:
      kss(sessID, inD, workD, outD)

  if data['notifications']["enable_gmail_notifications"] and not SILENCE_NOTIFICATIONS:
      try:
          import smtplib
          from email.mime.multipart import MIMEMultipart
          from email.mime.text import MIMEText
          mail_content = data['notifications']["message"]
          sender_address = data['notifications']["send_from"]
          sender_pass = data['notifications']["password"]
          receiver_address = data['notifications']["send_to"]
          message = MIMEMultipart()
          message['From'] = sender_address
          message['To'] = receiver_address
          message['Subject'] = 'KSS'
          message.attach(MIMEText(mail_content, 'plain'))
          session = smtplib.SMTP('smtp.gmail.com', 587)
          session.starttls()
          session.login(sender_address, sender_pass)
          text = message.as_string()
          session.sendmail(sender_address, receiver_address, text)
          session.quit()
          print('Mail Sent')
      except:
          print(" ! Failed to send email notification!  check your options file.")

  b = datetime.datetime.now()
  print(f"total time: {b-a}")
