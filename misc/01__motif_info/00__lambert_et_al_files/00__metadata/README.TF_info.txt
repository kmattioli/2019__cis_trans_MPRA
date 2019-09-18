This file contains the official list of human TFs from Lambert et al (PMID:29425488), along
with all associated information.  Candidate proteins were manually examined by a panel
of experts based on available data.  Proteins with experimentally demonstrated DNA
binding specificity were considered TFs.  Other proteins, such as co-factors and RNA 
binding proteins, were classified as non-TFs.  All proteins (both TFs and non-TFs) are
contained in the database, along with the associated evidence.

See our accompanying publication (PMID:29425488) and website (http://humantfs.ccbr.utoronto.ca/) 
for further details. Accompanying DNA binding motifs can be obtained from the Cis-BP
web server (http://cisbp.ccbr.utoronto.ca/).  Questions and comments can be directed to 
Matt Weirauch (Matthew.Weirauch@cchmc.org).



Database fields:


ENSEMBL ID - Official Ensembl gene ID.

HGNCsymbol - Official gene name.

DBD - DNA binding domains contained in the protein(s).

Is TF? - Is the protein a TF (i.e., does it bind DNA specifically?) (Yes/No).

TF assessment - Assessment of binding activity. 
	"Known motif" = A DNA motif is currently available.
	"Likely to be sequence specific" = No motif available, but evidence for DNA binding.
	"ssDNA/RNA binding" = Likely to be a single-stranded DNA or RNA binding protein.
	"Unlikely to be sequence specific" = Lack of strong evidence for sequence-specific DNA binding.

Binding mode - Mode of interacting with DNA.
	"Monomer or Homomultimer" = Binds DNA as a monomer, homodimer, homotrimer, etc. Some
		of these can also bind as heteromers. Proteins were classified in this 
		category if they are capable of binding DNA without the aid of other proteins.
	"Obligate heteromer" = Can only bind as a heteromer.
	"Low specificity DNA binding" = Binds DNA with little or no sequence specificity.
	"Not a DNA binding protein" = You guessed it.

Motif status - Current status of motif availability.
	"High throughput in vitro" = In vitro-derived motif (e.g. PBM, HT-SELEX, B1H) available.
	"100 perc ID - in vitro" = In vitro-derived motif available for another TF with an 
		identical DNA binding domain amino acid sequence. (So, the motif is 
		essentially available).
	"In vivo/Misc source" = Motif is only available from an in vivo (e.g. ChIP-seq) or
		low-throughput (e.g., SELEX) source.
	"No motif" = you guessed it again.

Final Notes - Final notes, automatically generated.  So, this column uses a controlled 
	vocabulary, suitable for automated classification/analysis.

Final Comments - Final comments, manually entered.  Summary comments combined and curated
	from the original reviewer comments.  So, this is all free text, human-readable only.

Interpro ID(s) - Interpro IDs for DBDs (semicolon-delimited).

EntrezGene ID - Entrez Gene ID, when available.

EntrezGene Description - Entrez Gene Description, when available.

PDB ID - Protein Data Bank ID (for structures of the protein or DBD in complex with DNA), 
	when available.

TF tested by HT-SELEX? - Has the protein been tested for DNA binding in a HT-SELEX assay in
	the Taipale lab?
	"DBD" = Tested using a construct with the DBD only
	"Full" = Tested using a construct with the full-length protein
	"not tested" = You guessed it yet again

TF tested by PBM? - Has the protein been tested for DNA binding in a PBM assay?

Conditional Binding Requirements - Notes on requirements for binding (e.g., requires post-
	translational modifications).  Note - this column is not comprehensive!

Original Comments - Original comments provided by the primary reviewer of the protein.

Vaquerizas 2009 TF classification - Classification provided by the Vaquerizas 2009 paper.
	"a" = Has direct evidence of TF function.
	"b" = Has evidence for an orthologous TF.
	"c" = Contains likely DBDs, but has no functional evidence.
	"x" = Is an unlikely TF such as predicted gene, genes with likely non-specific DBDs 
		or that have function outside transcription.
	"other" =  Protein that lacks clear DBDs that was curated from external sources.

CisBP considers it as a TF? - Is the protein available in the CisBP database (build 1.02)?

TFCat Classification - Does the TFCat web site classify the protein as a TF?

Is a GO TF - Does GO (Gene Ontology) classify the protein as a TF?

Initial assessment - Initial assessment provided by curators.

Curator 1 - Name of curator 1.

Curator 2 - Name of curator 2.

TFclass considers it as a TF? - Does TFclass consider the protein to be a TF?

Go Evidence - Evidence from GO supporting this protein being a TF.

Pfam Domains (By ENSP ID) - List of Pfam Domains contained in the protein.  Format ('#'-
	delimited): Ensembl Protein ID, Ensembl Gene ID, Ensembl Transcript ID, Pfam domains.
