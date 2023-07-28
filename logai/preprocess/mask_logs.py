import re


class MaskLogLine(str):

  def __init__(self, log_line: str):
    self.log_line = log_line
    self.patterns = {
      r'\b(?:\d{1,3}\.){3}\d{1,3}\b[:\d]*': "IP_ADDR",
      r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})": "MAC_ADDR",
      r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}": "UUID",
      r"https?:\/\/\S+": "URL",
      r"\/\S+/\S+": "FILE_PATH",
      r"[vV]ersion[: ]+(\S+)": "VERSION",
      r"[pP]ort (\d+)": "PORT_NO",
      r"[pP]id (\d+)": "PID",
      r"[pP]arent (\d+)": "PARENT PID",
      r"[cC]hild (\d+)": "CHILD PID",
      r"[pP]rocess id (\d+)": "PROCESS ID",
      r"score[sS) ,]+are([, \d+]+)": "SCORES",
      r"[iI][dD][ =:]\d+": " ID",
      r"[:,\t=()]+": " "
      # Add more patterns and their replacements as needed
      # Example:
      # r'pattern': 'replacement'}
    }

  def mask_log_line_efficient_way(self, log_line):
    # Perform the replacements in one pass using a callback function
    def repl_func(match):
      matched_text = match.group(0)
      for pattern, replacement in self.patterns.items():
        if re.match(pattern, matched_text):
          return replacement
      return matched_text  # Return the original text if no match is found

    return re.sub('|'.join(self.patterns.keys()), repl_func, log_line)

  def mask_ip(self):
    pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    replacement = "IP_ADDR"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_mac_address(self):
    pattern = r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})"
    replacement = "MAC_ADDR"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_pid(self):
    pattern = r"[pP]rocess id \d+"
    replacement = "process id PID"
    self.log_line = re.sub(pattern, replacement, self.log_line)

    pattern = r"[cC]hild \d+"
    replacement = "child PID"
    self.log_line = re.sub(pattern, replacement, self.log_line)

    pattern = r"[pP]arent \d+"
    replacement = "parent PID"
    self.log_line = re.sub(pattern, replacement, self.log_line)

    pattern = r"[pP]id \d+"
    replacement = "pid PID"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_port(self):
    pattern = r"[pP]ort \d+"
    replacement = "port PORT_NUM"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_version(self):
    pattern = r"[vV]ersion: \S+"
    replacement = "version: VERSION"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_file_dir_path(self):
    pattern = r"\/\S+/\S+"
    replacement = "FILE_DIR_PATH"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_url(self):
    pattern = r"https?:\/\/\S+"
    replacement = "URL"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_uuid(self):
    pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    replacement = "UUID"
    self.log_line = re.sub(pattern, replacement, self.log_line)

  def mask_log_line(self):
    masking_functions_list = [self.mask_ip, self.mask_mac_address, self.mask_port,
                              self.mask_pid, self.mask_url, self.mask_file_dir_path,
                              self.mask_uuid, self.mask_version]
    for mask_func in masking_functions_list:
      mask_func()


if __name__ == "__main__":
  logline = "Transport error reported by bottom half while trying to send RPC with rpc_id=8444787026930177039, detail=Http connection error.Node 10.23.52.53:23323 has been down (bec)ause scores are 23 53 32. pid 23 was responsible\t for this creating nusinace=on port 23. Version 3.5.6 was responsible in path /dfd/dfd/df/df"
  masker = MaskLogLine(logline)
  nlogline = masker.mask_log_line_efficient_way(logline)
  print("Log:%s\nModified Log:%s" % (logline, nlogline))