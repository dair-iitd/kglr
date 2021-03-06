This repository contains code and data accompanying the paper: Prachi Jain and Mausam. Knowledge-Guided Linguistic Rewrites for Inference Rule Verification. in: Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics – Human Language Technologies (NAACL HLT). 2016. [[pdf]](http://homes.cs.washington.edu/~mausam/papers/naacl16b.pdf)

"How to Run":

Details :

    [kglr_settings.py] ( KGLR/kglr_settings.py ): Set up all paths specified in the file.

    [ablation.txt] ( in_files/ablation.txt ): Sample input rule file (Add your own)

    [kglr_core.py] ( KGLR/kglr_core.py ): All core rule implementaion are in this file
    
    [kglr_main.py] ( KGLR/kglr_main.py ): Run python kglr_main.py
    
    [generate_output_file.sh] ( out_files/generate_output_file.sh ): After running python kglr_main.py, run ./generate_output_file.sh

    [outfile.txt] ( out_files/outfile.txt ): System output will be stored in this file

    [thesaurus.sql] You need thesaurus in the following format:
    -- SQL Dump
    SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
    SET time_zone = "+00:00"
    /*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
    /*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
    /*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
    /*!40101 SET NAMES utf8 */;
    --
    -- Database: `thesaurus`
    --
    -- --------------------------------------------------------
    --
    -- Table structure for table `antonyms`
    --
    CREATE TABLE IF NOT EXISTS `antonyms` (
    `Word` varchar(767) NOT NULL DEFAULT '',
    `Sense` varchar(767) NOT NULL DEFAULT '',
    `Word_sense_antonyms` varchar(767) DEFAULT NULL,
    PRIMARY KEY (`Word`,`Sense`),
    UNIQUE KEY `all3` (`Word`,`Sense`,`Word_sense_antonyms`)
    ) ENGINE=InnoDB DEFAULT CHARSET=latin1;
    -- --------------------------------------------------------
    --
    -- Table structure for table `synonyms`
    --
    CREATE TABLE IF NOT EXISTS `synonyms` (
    `Word` varchar(767) NOT NULL DEFAULT '',
    `Sense` varchar(767) NOT NULL DEFAULT '',
    `Word_sense_synonyms` varchar(767) DEFAULT NULL,
    PRIMARY KEY (`Word`,`Sense`),
    UNIQUE KEY `all3` (`Word`,`Sense`,`Word_sense_synonyms`)
    ) ENGINE=InnoDB DEFAULT CHARSET=latin1;
    -- --------------------------------------------------------
    --
    -- Table structure for table `syn_of_syn`
    --
    CREATE TABLE IF NOT EXISTS `syn_of_syn` (
    `Word` varchar(767) NOT NULL DEFAULT '',
    `Sense` varchar(767) NOT NULL DEFAULT '',
    `Word_sense_syn_of_syn` varchar(767) DEFAULT NULL,
    PRIMARY KEY (`Word`,`Sense`),
    UNIQUE KEY `all3` (`Word`,`Sense`,`Word_sense_syn_of_syn`)
    ) ENGINE=InnoDB DEFAULT CHARSET=latin1;
    /*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
    /*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
    /*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
    --
    -- Sample Entries
    INSERT INTO `antonyms` (`Word`, `Sense`, `Word_sense_antonyms`) VALUES ('able-bodied', 'adj:physically strong and capable', 'weak;delicate;infirm');
    INSERT INTO `syn_of_syn` (`Word`, `Sense`, `Word_sense_syn_of_syn`) VALUES ('zipped', 'verb:scurry:move along swiftly', 'rip;dash;skim;sprint;hop along;whirl;zip;scutter;scamper;shoot;run;step along;dart;bustle;dust;hurry;fly;rush;whisk;tear;scurry;scud;move along swiftly;race;scoot;barrel;hasten;scuttle');
    INSERT INTO `synonyms` (`Word`, `Sense`, `Word_sense_synonyms`) VALUES ('secure', 'adj:fastened:stable', 'tenacious;adjusted;set;bound;anchored;staunch;solid as a rock;safe and sound;buttoned down;immovable;nailed;fast;tight;stable;firm;sure;locked;strong;sound;fastened;solid;fortified;steady;iron;fixed');
