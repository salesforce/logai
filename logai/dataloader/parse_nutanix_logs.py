import re
from collections import OrderedDict


class ParseNutanixLogs(object):
  """
  class which contains logic to automatically detect iterator
  """
  # TODO: add body too in regex variable and return
  ITER_REGEX_MAP = OrderedDict([
    ("py", [
      # 2018-05-16 12:28:48 INFO 56294656 minerva_ha.py:2170 Start taking over
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2})\s*(?P<level>[A-Z]+)"
                 r"\s*\d+\s*(?P<source_file>[\S]+):"
                 r"\d+\s*(?P<body>[\S\s]+)"),

      # 2018-05-16 12:28:48 INFO minerva_ha.py:2170 Start taking over
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2})\s*(?P<level>[A-Z]+)"
                 r"\s*\s*(?P<source_file>[\S]+):"
                 r"\d+\s*(?P<body>[\S\s]+)"),               

      # 2020-01-28 12:57:44,123Z INFO proxy_client.py:146 Forwarding 1 request
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2})\S+\s*"
                 r"(?P<level>[A-Z]+)\s*"
                 r"(?P<source_file>[\S]+):\d+\s*(?P<body>[\S\s]+)"),

      # 2020-12-11 12:29:28,713 INFO 28611728 minerva_common_utils.py:89 ZK
      # session establishment complete, sessionId=0x1762703e2e12be2, negotiated
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2})\S+\s*(?P<level>[A-Z]+)"
                 r"\s*\d+\s*(?P<source_file>[\S]+)"
                 r":\d+\s*(?P<body>[\S\s]+)")
    ]),
    ("java", [
      # INFO [FlushWriter:519] 2017-12-31 12:13:39,484 Memtable.java
      # (line 337) Completed flushing
      re.compile(r"(?P<level>[A-Z]+)\s*\[[\S+:\d]+\]\s*"
                 r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2}),\d{3}[Z]*\s*"
                 r"(?P<source_file>[\S]+)\s*\(line\s*\d+\)\s*"
                 r"(?P<body>[\s\S]+)"),
      # INFO  2019-02-25 15:05:11,535 ZeusConfigCacheService
      # adapter.service.ZeusConfigCacheService.startUp:96
      # Starting service, ZeusConfigCacheService
      re.compile(r"(?P<level>[A-Z]+)\s*(?P<date>\d{4}-\d{2}-\d{2}\s*"
                 r"\d{2}\:\d{2}\:\d{2}),\d{3}[Z]*\s*\S+\s*(?P<source_file>\S+)"
                 r":\d*\s*(?P<body>.*)")
    ]),
    ("messages", [
      # 2021-05-28T04:28:40.413144+02:00 NTNX-XX-A-CVM kernel:
      # [106064.317871] IPTables Packet Dropped
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d*[+-]\d{2}:\d{2}\s*\S+\s*"
                 r"(?P<source_file>kernel):\s*\[\d+\.\d+\]\s*(?P<body>[\S\s]+)"),
      # 2019-12-27T17:18:01.482487-05:00 NTNX-16SM6B490273-A-CVM systemd[1]:
      # Started Session 881311 of user nutanix.
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d*[+-]\d{2}:\d{2}\s*"
                 r"\S+\s*(?P<source_file>\S+)\[\d*\]:(?P<body>.*)")
    ]),
    ("msp", [
      #0                                1           2
      #time="2019-12-12T18:47:06-08:00" level=error msg="Hit error in retry number 0: Fetching IAM endpoint"
      re.compile(r"time=\"(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\:\d{2})\S+\"\s*level=(?P<level>[a-zA-Z]+)"
                 r"\s*(?P<body>[\S\s]+)")

      #0                                1           2
      #time="2019-12-12T18:47:06Z" level=error msg="Hit error in retry number 0: Fetching IAM endpoint"
                      
    ]),
    ("ovsvswitchd", [
      #         0                 1       2             3         4
      #2022-05-19T11:19:35.015Z|06007|ovs_rcu(urcu6)|WARN|blocked 2000 ms waiting for main to quiesce
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\:\d{2}).\S+\|\d+\|(?P<source_file>\S+)\|(?P<level>[A-Z]+)\|"
                 r"(?P<body>[\S\s]+)")
    ]),
    ("samba", [    
      #[2020/12/01 17:10:58.839972,  2, pid=25838, class=winbind] ../source3/winbindd/winbindd_util.c:317(add_trusted_domain_from_tdc) Added domain CHILD2 child2.afs.minerva.com S-1-5-21-2181377586-1363663071-3087203698
      re.compile(r"\[(?P<date>\d{4}\/\d{2}\/\d{2}\s*\d{2}:\d{2}\:\d{2}).*\]\s*(?P<source_file>\S+)[:(]\S+\s*"
                 r"(?P<body>[\S\s]+)"),

      #2022-05-10 21:34:57.025514Z  1, 182821, spnego.c:618 gensec_spnego_create_negTokenInit SPNEGO(gse_krb5) creating NEG_TOKEN_INIT failed: NT_STATUS_INTERNAL_ERROR
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}\:\d{2}).*,\s*\d+,\s*(?P<source_file>\S+):\S+\s*"
                 r"(?P<body>[\S\s]+)")                 
    ]),
    ("cpp", [
      # I0318 12:33:24.851739 40435 init_nutanix.cc:352] Started child 40435 that is being monitored by parent 40434
      # I20220318 12:33:24.851739Z 40435 init_nutanix.cc:352] Started child 40435 that is being monitored by parent 40434
      re.compile(r"(?P<level>I|W|E|F)(?P<date>\d+\s*\d{2}:\d{2}:\d{2})\S+\s*\d+\s*"
                 r"(?P<source_file>[\S]+):\d+]\s*(?P<body>.*)")
    ]),
    ("catalina", [
      # Jul 20, 2018 4:28:11 AM org.apache.catalina.core.StandardService
      # startInternal
      re.compile(r"[A-Za-z]+\s*\d{2},\s*\d{4}\s*\d+:\d+:\d+\s*[AP]M\s*"
                 r"(?P<source_file>[\d\S]+)\s*(?P<body>.*)")
    ]),
    ("ganesha", [
      # 02/12/2020 21:14:18 : epoch 00000043 : ntnx-10-53-79-224-a-fsvm :
      # nfs-ganesha-21832[main] nfs_start :NFS STARTUP
      # :EVENT :             NFS SERVER INITIALIZED
      re.compile(r"(?P<date>\d{2}\/\d{2}\/\d{4}\s*\d{2}:\d{2}:\d{2})\s*:\s*epoch\s*\S+\s*:\s*"
                 r"[a-zA-Z0-9-]+\s*:\s*[a-zA-Z0-9-]+\[\S+\]\s*"
                 r"(?P<source_file>\S+)\s*:[\S\s\d]*:(?P<level>\S+)\s*:"
                 r"\s*(?P<body>.*)")
    ]),
    ("go", [
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2})\s*(?P<level>[A-Z]+)"
                 r"\s*\d+\s*(?P<source_file>[\S]+\.[a-z]*):"
                 r"\d+\s*(?P<body>[\S\s]+)")
    ]),    
    ("prism", [
      # ERROR 2017-03-13 19:25:31,265 http-nio-127.0.0.1-9081-exec-4
      # prism.aop.RequestInterceptor.invoke:140 Throwing exception...
      re.compile(r"(?P<level>[A-Z]+)\s*(?P<date>\d{4}-\d{2}-\d{2}"
                 r"\s*\d{2}\:\d{2}\:\d{2}),\d{3}[Z]*\s*(?P<thread>\S+)\s*"
                 r"(?P<source_file>\S+)\:\d*\s*(?P<body>.*)"),
      # INFO  2020-03-18 12:29:08,066 ZeusConfigCacheService []
      # adapter.service.ZeusConfigCacheService.startUp:96
      # Starting service, ZeusConfigCacheService
      re.compile(r"(?P<level>[A-Z]+)\s*(?P<date>\d{4}-\d{2}-\d{2}\s*"
                 r"\d{2}\:\d{2}\:\d{2}),\d{3}[Z]*\s*(?P<thread>\S+)\s*\[\S*\]\s*"
                 r"(?P<source_file>\S+)\:\d*\s(?P<body>.*)")
    ]),
    ("zookeeper", [
      # 2021-01-26 17:40:29,123Z - INFO
      # [NIOServerCxn.Factory:0.0.0.0/0.0.0.0:9876:NIOServerCnxnFactory@228]
      # - Accepted socket connection from /10.48.64.223:48212
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2})\S+\s*-\s*"
                 r"(?P<level>[A-Z]+)\s*\S+\:\d+\:"
                 r"(?P<source_file>\S+)@\d+\]\s*-\s*(?P<body>.*)"),

      # 2021-01-08 11:17:02,875Z - INFO  [Thread-8:NIOServerCnxn@1027] - Closed 
      # socket connection for client /10.136.106.164:51242 (no session established for client)
      re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}\s*\d{2}\:\d{2}\:\d{2})\S+\s*-\s*"
                 r"(?P<level>[A-Z]+)\s*\S+\:"
                 r"(?P<source_file>\S+)@\d+\]\s*-\s*(?P<body>.*)")                 
    ])

  ])
  COMPONENT_DEFAULTS = {
    "messages": {
      "level": "I"
    },
    "catalina": {
      "level": "I"
    },
    "generic": {
      "level": "",
      "source_file": ""
    }
  }

  def __init__(self):
    self.log_text = None

  def _find_iterator_from_log(self):
    for iterator, regexes in self.ITER_REGEX_MAP.items():
      for regex_raw in regexes:
        regex = re.compile(regex_raw)
        match = regex_raw.search(self.log_text)
        if match:
          return iterator, match.groupdict()

    return "generic", {}

  def extract_log_details(self, log_text):
    """
    main api which takes in input of log text and parses it for details
    """
    # example:
    #   log_text = "2018-05-16 12:28:48 INFO 56294656 minerva_ha.py:2170 Start
    #               taking over"
    self.log_text = str(log_text)
    # print(log_text)
    # example:
    #   iterator = "py"
    #   log_details = {
    #     "level": "INFO",
    #     "source_file": "minerva_ha.py"
    #   }
    iterator, log_details = self._find_iterator_from_log()
    # print(iterator)
    # print("\n")
    # print(log_details)
    if iterator == "zookeeper":
      if "source_file" in log_details:
        log_details["source_file"] = "%s.java" % log_details["source_file"]
    log_details.update(self.COMPONENT_DEFAULTS.get(iterator, {}))
    self.log_text = None
    return iterator, log_details


if __name__ == "__main__":
  parser = ParseNutanixLogs()
  print(parser.extract_log_details(["2021-01-26 17:40:29,123Z - "
                                         "INFO NIOServerCxn.Factory:0.0.0.0/"
                                         "0.0.0.0:9876:NIOServerCnxnFactory"
                                         "@228] - Accepted socket connection "
                                         "from /10.48.64.223:48212"]))
