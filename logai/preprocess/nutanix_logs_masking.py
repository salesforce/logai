import re


class MaskLogLine(str):

  def __init__(self, log_line: str):
    self.log_line = log_line

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
