import re
import math
from urllib.parse import urlparse
import tldextract

def randomness(text):
  #Calculate the randomess of string
  if not text:
    return 0
  frequency = {}
  for ch in text:
    frequency[ch]=frequency.get(ch,0)+1
    length=len(text)
    entropy=0
  for count in frequency.values():
      prblty=count/length
      entropy=entropy-prblty*math.log2(prblty)
  return entropy

def ip_addr_check(url):
   pattern=re.compile(
      r'(([01]?\d\d|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[05])'
   )
   return 1 if pattern.search(url) else 0

def extract_features(url):
    features={}

    try:
       parsed=urlparse(url)
       extracted=tldextract.extract(url)
    except Exception:
       return {f"feature_{i}": 0 for i in range(22)}
    
    scheme=parsed.scheme
    netloc=parsed.netloc
    path=parsed.path
    query=parsed.query
    subdomain=extracted.subdomain
    domain=extracted.domain
    suffix=extracted.suffix

    compl_domain=netloc

    #Phishing URLS are generally longer than most URLs
    features['url_length']=len(url)
    features['domain_length']=len(domain)
    features['path_length']=len(path)
    features['query_length']=len(query)

    #Counters to track Unusual counts of special characters
    features['num_dots']=url.count('.')
    features['num_hyphens']=url.count('-')
    features['num_underscores']=url.count('_')
    features['num_slashes']=url.count('/')
    features['num_at']=url.count('@')
    features['num_question']=url.count('?')
    features['num_equals']=url.count('=')
    features['num_ampersand']=url.count('&')
    features['num_digits']=sum(c.isdigit() for c in url)
    features['num_subdomains']=len(subdomain.split('.')) if subdomain else 0

    features['uses_https']=1 if scheme=='https' else 0
    features['has_ip']=ip_addr_check(url)
    suspicious_tlds={'tk','ml','cf','ga','gq','pw','top','xyz','click','link'}
    features['suspicious_tld']=1 if suffix.lower() in suspicious_tlds else 0
    features['has_port']=1 if ':' in netloc and not netloc.endswith(':80') and not netloc.endswith(':443') else 0
    features['has_double_slash']=1 if '//' in path else 0
    keywords=['login','verify','secure','account','update','confirm','bank','paypal','signin','password','credential','authenticate','webscr','ebayisapi']
    features['has_sus_keyword']=1 if any(kw in url.lower() for kw in keywords) else 0
    features['url_randomness']=round(randomness(url),4)
    features['domain_randomness']=round(randomness(domain),4)

    return features

# if __name__=="__main__":
#     test_url="http://example.com/login?user=abc&pass=123"
#     features=extract_features(test_url)
#     for key,value in features.items():
#         print(f"{key}: {value}")