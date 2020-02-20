all: grn.mor.hfst quc.gen.hfst

grn.lexc.hfst: grn.lexc
	hfst-lexc grn.lexc -o grn.lexc.hfst
grn.twol.hfst: grn.twol
	hfst-twolc grn.twol -o grn.twol.hfst
grn.gen.hfst: grn.lexc.hfst grn.twol.hfst
	hfst-compose-intersect -1 grn.lexc.hfst -2 grn.twol.hfst -o grn.gen.hfst
grn.mor.hfst: grn.gen.hfst
	hfst-invert grn.gen.hfst -o grn.mor.hfst


quc.lexc.hfst: quc.lexc
	hfst-lexc quc.lexc -o quc.lexc.hfst
quc.twol.hfst: quc.twol
	hfst-twolc quc.twol -o quc.twol.hfst
quc.gen.hfst: quc.lexc.hfst quc.twol.hfst
	hfst-compose-intersect -1 quc.lexc.hfst -2 quc.twol.hfst -o quc.gen.hfst
