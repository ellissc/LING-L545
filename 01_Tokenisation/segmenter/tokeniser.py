import sys, re

abbr = ["age.","alb.","alt.","alz.","ang.","ast.","atd.","atp.","aus.","bal.","bas.","btp.","buł.","bůn.","bůł.","cer.","cyg.","cyr.","dia.","dna.","dom.","dow.","dńi.","dům.","důł.","ery.","est.","etc.","fet.","fiń.","gag.","gal.","gas.","gaz.","gen.","grc.","gre.","gyn.","han.","haw.","hel.","hul.","ino.","int.","irl.","itd.","jak.","jap.","jid.","jod.","jům.","kat.","kaz.","kdb.","kjh.","kol.","kor.","krc.","krm.","kůu.","lad.","lap.","las.","lat.","lew.","lit.","los.","lot.","lud.","lŏt.","lůd.","min.","mln.","moc.","mon.","můn.","nah.","nar.","nep.","niy.","nov.","ntp.","okc.","opy.","paś.","per.","poj.","pol.","pop.","por.","pot.","rap.","ras.","ret.","rod.","rok.","ros.","roz.","rum.","rus.","rům.","sch.","sie.","skr.","sua.","szp.","szw.","sōm.","sůl.","tam.","tle.","tmj.","trb.","trl.","tum.","tur.","tyb.","typ.","tys.","tyś.","tyż.","tzn.","tzw.","tło.","uac.","uać.","ukr.","ulg.","uok.","uos.","uot.","uůn.","var.","wal.","woj.","wsi.","wym.","wyn.","wśi.","yng.","zał.","zaś.","zem.","zoc.","zol.","zou.","ông.","čes.","čos.","łac.","łać.","łeb.","ńid.","ńim.","ńym.","ńům.","ůkr.","ůng.","żył.","um.","r.","m.","f.","św.","zm."]

def tokenise(line, abbr):
	if line.strip() == '':
		return ''
	line = re.sub(r'([\(\)”?:!;])',r' \g<1> ', line) #splits off always-separating punctuation
	line = re.sub(r'([^0-9]),',r'\g<1> ,', line) #leaves commas if no number vefore
	line = re.sub(r',([^0-9])',r', \g<1>', line) #see slides, \g - group
	line = re.sub(r'  +',' ', line[:-1]) #collapse sequences of spaces to one space

	output = []
	for token in line.split(' '): #split off full stops not apart of abbrev
		if token == '':
			continue
		if token[-1] == '.' and token not in abbr:
			token = token[:-1] + ' .'
		output.append(token)
	return ' '.join(output)

line = sys.stdin.readline()
while line != '':
	print(tokenise(line.strip('\n'), abbr))
	line = sys.stdin.readline()
