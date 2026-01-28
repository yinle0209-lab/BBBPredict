import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import re
import os
import math
import numpy
import pandas
from collections import Counter
import csv
import glob

class FASTA(object):
    def __init__(self, file):
        self.file = file
        self.fasta_list = []
        self.number = 0
        self.encoding_array = numpy.array([])
        self.row = 0
        self.column = 0

        self.fasta_list = self.read_fasta(self.file)
        self.sub_fasta_list = [x[1:3] for x in self.fasta_list]
        self.number = len(self.fasta_list)

    def read_fasta(self, file):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.split('>')[1:]
        fasta_sequences = []
        for fasta in text:
            array = fasta.split('\n')
            header = array[0].split()[0]
            header_array = header.split('|')
            sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
            name = header_array[0]
            label = header_array[1]
            train = header_array[2]
            fasta_sequences.append([name, sequence, label, train])
        return fasta_sequences

    def save(self, FileName):
        file = self.encoding_array
        file_save = file[:, 1:]
        numpy.savetxt(FileName, file_save, fmt='%s', delimiter=',')


class AAC(FASTA):
    def __init__(self, file):
        super(AAC, self).__init__(file)

    def AAC(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        self.encoding_array = numpy.array([])
        header = ['Name', 'label']
        for i in AA:
            header.append(i)
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            occur = Counter(sequence)
            sum = len(sequence)
            for i in AA:
                code.append(occur[i] / sum)
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array[0]
        self.column = self.encoding_array[1]
        del encodings

class CKSAAP(FASTA):
    def __init__(self, file):
        super(CKSAAP, self).__init__(file)

    def CKSAAP(self):
        gap = 3
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        AA_Pairs = []
        self.encoding_array = numpy.array([])
        for i in AA:
            for j in AA:
                AA_Pairs.append(i + j)
        header = ['Name', 'label']
        for g in range(gap + 1):
            for i in AA_Pairs:
                header.append(i + '.gap' + str(g))
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            for g in range(gap + 1):
                occur = {}
                for pair in AA_Pairs:
                    occur[pair] = 0
                for i in range(len(sequence) - g - 1):
                    j = i + g + 1
                    occur[sequence[i] + sequence[j]] += 1
                for pair in AA_Pairs:
                    code.append(occur[pair] / (len(sequence) - g - 1))
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array[0]
        self.column = self.encoding_array[1]
        del encodings


class DDE(FASTA):
    def __init__(self, file):
        super(DDE, self).__init__(file)

    def DDE(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        AA_Pairs = []
        self.encoding_array = numpy.array([])
        for i in AA:
            for j in AA:
                AA_Pairs.append(i + j)
        header = ['Name', 'label'] + AA_Pairs
        encodings.append(header)
        Codons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                  'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2}

        Tm = []
        for i in AA_Pairs:
            Tm.append((Codons[i[0]] / 61) * (Codons[i[1]] / 61))

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            temp = []
            occur = {}
            for pair in AA_Pairs:
                occur[pair] = 0
            for i in range(len(sequence) - 1):
                occur[sequence[i] + sequence[i + 1]] += 1
            for pair in AA_Pairs:
                temp.append(occur[pair] / (len(sequence) - 1))

            Tv = []
            for i in range(len(Tm)):
                Tv.append(Tm[i] * (1 - Tm[i]) / (len(sequence) - 1))

            for i in range(len(temp)):
                temp[i] = (temp[i] - Tm[i]) / math.sqrt(Tv[i])
            code = code + temp
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array[0]
        self.column = self.encoding_array[1]
        del encodings


class APAAC(FASTA):
    def __init__(self, file):
        super(APAAC, self).__init__(file)

    def APAAC(self):
        lambdaValue = 2
        w = 0.05
        dataFile = r'PAAC.txt'

        with open(dataFile) as f:
            PAAC = f.readlines()

        AA = ''.join(PAAC[0].rstrip().split()[1:])
        dict = {}
        prop1 = []
        name = []
        encodings = []
        self.encoding_array = numpy.array([])
        header = ['Name', 'label']
        for i in range(len(AA)):
            dict[AA[i]] = i
        for i in range(1, len(PAAC) - 1):
            array = PAAC[i].rstrip().split() if PAAC[i].rstrip() != '' else None
            prop1.append([float(j) for j in array[1:]])
            name.append(array[0])
        for i in AA:
            header.append('Pc1.' + i)
        for j in range(1, lambdaValue + 1):
            for i in name:
                header.append('Pc2.' + i + '.' + str(j))
        encodings.append(header)

        prop2 = []
        for i in prop1:
            mean = sum(i) / 20
            den = math.sqrt(sum([(j - mean) ** 2 for j in i]) / 20)
            prop2.append([(j - mean) / den for j in i])

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            theta = []
            for n in range(1, lambdaValue + 1):
                for i in range(len(prop2)):
                    theta.append(sum([prop2[i][dict[sequence[j]]] * prop2[i][dict[sequence[j + n]]] for j in
                                      range(len(sequence) - n)]) / (len(sequence) - n))

            occur = {}
            for i in AA:
                occur[i] = sequence.count(i)
            for i in AA:
                code = code + [occur[i] / (1 + w * sum(theta))]
            for i in theta:
                code = code + [w * i / (1 + w * sum(theta))]
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

class ASDC(FASTA):
    def __init__(self, file):
        super(ASDC, self).__init__(file)

    def ASDC(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        AA_Pairs = []
        self.encoding_array = numpy.array([])
        for i in AA:
            for j in AA:
                AA_Pairs.append(i + j)
        header = ['Name', 'label'] + AA_Pairs
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            occur = {}
            sum = 0
            for i in AA_Pairs:
                occur[i] = 0
            for i in range(len(sequence) - 1):
                for j in range(i + 1, len(sequence)):
                    occur[sequence[i] + sequence[j]] += 1
                    sum += 1
            for i in AA_Pairs:
                code.append(occur[i] / sum)
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class CTDC(FASTA):
    def __init__(self, file):
        super(CTDC, self).__init__(file)

    def Count_C(self, s, t):
        sum = 0
        for i in s:
            sum = sum + t.count(i)
        return sum

    def CTDC(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }
        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
        self.encoding_array = numpy.array([])
        encodings = []
        header = ['Name', 'label']
        for i in property:
            for j in range(1, len(groups) + 1):
                header.append(i + '.G' + str(j))
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            for i in property:
                c1 = self.Count_C(group1[i], sequence) / len(sequence)
                c2 = self.Count_C(group2[i], sequence) / len(sequence)
                c3 = 1 - c1 - c2
                code = code + [c1, c2, c3]
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class CTDT(FASTA):
    def __init__(self, file):
        super(CTDT, self).__init__(file)

    def CTDT(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }
        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
        self.encoding_array = numpy.array([])
        encodings = []
        header = ['Name', 'label']
        for i in property:
            for j in ('Tr1221', 'Tr1331', 'Tr2332'):
                header.append(i + '.' + j)
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            for i in range(len(sequence) - 1):
                AA_Pair = [sequence[i:i + 2]]
            for i in property:
                c1221 = c1331 = c2332 = 0
                for AA in AA_Pair:
                    if ((AA[0] in group1[i]) and (AA[1] in group2[i])):
                        c1221 += 1
                        continue
                    if ((AA[0] in group2[i]) and (AA[1] in group1[i])):
                        c1221 += 1
                        continue
                    if ((AA[0] in group1[i]) and (AA[1] in group3[i])):
                        c1331 += 1
                        continue
                    if ((AA[0] in group3[i]) and (AA[1] in group1[i])):
                        c1331 += 1
                        continue
                    if ((AA[0] in group2[i]) and (AA[1] in group3[i])):
                        c2332 += 1
                        continue
                    if ((AA[0] in group3[i]) and (AA[1] in group2[i])):
                        c2332 += 1
                        continue
                code = code + [c1221 / len(AA_Pair), c1331 / len(AA_Pair), c2332 / len(AA_Pair)]
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class CTDD(FASTA):
    def __init__(self, file):
        super(CTDD, self).__init__(file)

    def Count_D(self, group, s):
        sum = 0
        for i in s:
            if i in group:
                sum += 1
        node = [1, math.floor(0.25 * sum), math.floor(0.5 * sum), math.floor(0.75 * sum), sum]
        node = [i if i >= 1 else 1 for i in node]

        code = []
        for n in node:
            sum = 0
            for i in range(len(s)):
                if s[i] in group:
                    sum += 1
                    if sum == n:
                        code.append((i + 1) / len(s) * 100)
                        break
            if sum == 0:
                code.append(0)
        return code

    def CTDD(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }
        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
        self.encoding_array = numpy.array([])
        encodings = []
        header = ['Name', 'label']
        for i in property:
            for j in ('1', '2', '3'):
                for k in ['0', '25', '50', '75', '100']:
                    header.append(i + '.' + j + '.residue' + k)
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            for i in property:
                code = code + self.Count_D(group1[i], sequence) + self.Count_D(group2[i], sequence) + self.Count_D(
                    group3[i], sequence)
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class QSO(FASTA):
    def __init__(self, file):
        super(QSO, self).__init__(file)

    def QSO(self):
        nlag = 2
        w = 0.05
        dataFile_S = r'Schneider-Wrede.txt'
        dataFile_G = r'Grantham.txt'
        AA1 = 'ACDEFGHIKLMNPQRSTVWY'
        AA2 = 'ARNDCQEGHILKMFPSTWYV'
        self.encoding_array = numpy.array([])
        dict_S = {}
        dict_G = {}
        for i in range(len(AA1)):
            dict_S[AA1[i]] = i
            dict_G[AA2[i]] = i

        with open(dataFile_S) as f:
            text = f.readlines()[1:]
        distance_S = []
        for i in text:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            distance_S.append(array)
        distance_S = numpy.array(
            [float(distance_S[i][j]) for i in range(len(distance_S)) for j in range(len(distance_S[i]))]).reshape(
            (20, 20))

        with open(dataFile_G) as f:
            text = f.readlines()[1:]
        distance_G = []
        for i in text:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            distance_G.append(array)
        distance_G = numpy.array(
            [float(distance_G[i][j]) for i in range(len(distance_G)) for j in range(len(distance_G[i]))]).reshape(
            (20, 20))

        encodings = []
        header = ['Name', 'label']
        for a in AA1:
            header.append('Schneider-Wrede.Xr.' + a)
        for a in AA2:
            header.append('Grantham.Xr.' + a)
        for n in range(1, nlag + 1):
            header.append('Schneider-Wrede.Xd.' + str(n))
        for n in range(1, nlag + 1):
            header.append('Grantham.Xd.' + str(n))
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            array_S = []
            array_G = []
            for n in range(1, nlag + 1):
                array_S.append(sum([distance_S[dict_S[sequence[i]]][dict_S[sequence[i + n]]] ** 2 for i in
                                    range(len(sequence) - n)]))
                array_G.append(sum([distance_G[dict_G[sequence[i]]][dict_G[sequence[i + n]]] ** 2 for i in
                                    range(len(sequence) - n)]))

            occur = {}
            for a in AA2:
                occur[a] = sequence.count(a)
            for a in AA2:
                code.append(occur[a] / (1 + w * sum(array_S)))
            for a in AA2:
                code.append(occur[a] / (1 + w * sum(array_G)))
            for i in array_S:
                code.append((w * i) / (1 + w * sum(array_S)))
            for i in array_G:
                code.append((w * i) / (1 + w * sum(array_G)))
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

##########新特征#########
class DPC(FASTA):
    def __init__(self, file):
        super(DPC, self).__init__(file)

    def DPC(self):
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        dipeptides = [a+b for a in aa_list for b in aa_list]  # 20x20 dipeptides

        encodings = []
        header = ['Name', 'Label'] + dipeptides  # Add dipeptides as features
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            total_length = len(sequence)
            for dipeptide in dipeptides:
                count = sum([1 for i in range(total_length - 1) if sequence[i:i + 2] == dipeptide])
                code.append(count / total_length)  # Normalize by total length of the sequence
            encodings.append(code)

        self.encoding_array = np.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class TPC(FASTA):
    def __init__(self, file):
        super(TPC, self).__init__(file)

    def TPC(self):
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        tripeptides = [a+b+c for a in aa_list for b in aa_list for c in aa_list]  # 20x20x20 tripeptides

        encodings = []
        header = ['Name', 'Label'] + tripeptides  # Add tripeptides as features
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            total_length = len(sequence)
            for tripeptide in tripeptides:
                count = sum([1 for i in range(total_length - 2) if sequence[i:i + 3] == tripeptide])
                code.append(count / total_length)  # Normalize by total length of the sequence
            encodings.append(code)

        self.encoding_array = np.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings



class SE(FASTA):
    def __init__(self, file):
        super(SE, self).__init__(file)

    def SE(self):
        encodings = []
        header = ['Name', 'Label', 'SequenceEntropy']
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            aa_counts = {aa: sequence.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
            total_length = len(sequence)
            entropy = -sum((count / total_length) * math.log(count / total_length, 2) for count in aa_counts.values() if count > 0)
            code.append(entropy)
            encodings.append(code)

        self.encoding_array = np.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class SER(FASTA):
    def __init__(self, file):
        super(SER, self).__init__(file)

    def SER(self):
        encodings = []
        header = ['Name', 'Label', 'SequenceRepeat']
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            repeats = sum(1 for i in range(len(sequence) - 1) if sequence[i] == sequence[i + 1])
            code.append(repeats)
            encodings.append(code)

        self.encoding_array = np.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class SEP(FASTA):
    def __init__(self, file):
        super(SEP, self).__init__(file)

    def SEP(self, n=3):
        encodings = []
        header = ['Name', 'Label'] + [f"Pattern_{i}" for i in range(1, n+1)]  # Patterns of length n
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            total_length = len(sequence)
            for i in range(1, n+1):
                patterns = [sequence[j:j+i] for j in range(total_length - i + 1)]
                code.append(len(set(patterns)) / total_length)  # Normalize by total length of the sequence
            encodings.append(code)

        self.encoding_array = np.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings


class SOCN(FASTA):
    def __init__(self, file):
        super(SOCN, self).__init__(file)

    def SOCN(self):
        encodings = []
        header = ['Name', 'Label', 'SOCN']
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            socn_value = sum((ord(sequence[i]) - ord('A') + 1) * (ord(sequence[i+1]) - ord('A') + 1) for i in range(len(sequence) - 1))
            code.append(socn_value)
            encodings.append(code)

        self.encoding_array = np.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

#########新特征结束#######
class calculate_molecular_descriptors(FASTA):
    def __init__(self, file):
        super(calculate_molecular_descriptors, self).__init__(file)

    def calculate_molecular_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # List of 33 molecular descriptors
        descriptors = [
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            Descriptors.MaxAbsEStateIndex(mol),
            Descriptors.MinAbsEStateIndex(mol),
            Descriptors.qed(mol),
            Descriptors.MolWt(mol),
            Descriptors.HeavyAtomMolWt(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.MinAbsPartialCharge(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            Descriptors.NumSaturatedHeterocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumSaturatedCarbocycles(mol),
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.FractionCSP3(mol),
        ]

        return np.array(descriptors)

    def calculate(self):
        sequences = self.sub_fasta_list.iloc[:, 0].values
        labels = self.sub_fasta_list.iloc[:, 1].values

        # Initialize descriptor matrix (33 features per peptide)
        X = np.zeros((len(sequences), 33))

        for i, sequence in enumerate(sequences):
            peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(sequence))
            X[i] = self.calculate_molecular_descriptors(peptide_smiles)

        # Convert descriptors to DataFrame
        descriptors_df = pd.DataFrame(X, columns=[
            "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
            "QED", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
            "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge",
            "MinAbsPartialCharge", "BalabanJ", "BertzCT", "MolLogP", "MolMR",
            "HeavyAtomCount", "NHOHCount", "NumHDonors", "NumHAcceptors",
            "NumRotatableBonds", "NumHeteroatoms", "NumAromaticRings", "NumSaturatedRings",
            "NumAliphaticRings", "NumAromaticHeterocycles", "NumSaturatedHeterocycles",
            "NumAliphaticHeterocycles", "NumAromaticCarbocycles", "NumSaturatedCarbocycles",
            "NumAliphaticCarbocycles", "FractionCSP3"
        ])

        # Add peptide sequences and labels
        descriptors_df.insert(0, 'Peptide', sequences)
        descriptors_df.insert(1, 'Label', labels)

        return  self.descriptors_df

    def MolecularDescriptors(self):
        encodings = []
        header = ['Name', 'Label']

        # Add descriptor names to the header
        header.extend([
            "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
            "QED", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
            "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge",
            "MinAbsPartialCharge", "BalabanJ", "BertzCT", "MolLogP", "MolMR",
            "HeavyAtomCount", "NHOHCount", "NumHDonors", "NumHAcceptors",
            "NumRotatableBonds", "NumHeteroatoms", "NumAromaticRings", "NumSaturatedRings",
            "NumAliphaticRings", "NumAromaticHeterocycles", "NumSaturatedHeterocycles",
            "NumAliphaticHeterocycles", "NumAromaticCarbocycles", "NumSaturatedCarbocycles",
            "NumAliphaticCarbocycles", "FractionCSP3"
        ])
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]

            # Convert the peptide sequence to SMILES and calculate descriptors
            peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(sequence))
            descriptors = self.calculate_molecular_descriptors(peptide_smiles)

            if descriptors is not None:
                code.extend(descriptors)
            else:
                code.extend([None] * 33)  # Add None if the SMILES conversion fails

            encodings.append(code)

        self.encoding_array = np.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]



def FeatureExtraction(filename, output_path):
    # 1. 创建输出文件夹（如果不存在）
    # output_path = r'E:\Augur\RF\features'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    # 2. 创建特征提取对象
    f_AAC = AAC(filename)
    f_AAC.AAC()

    f_CKSAAP = CKSAAP(filename)
    f_CKSAAP.CKSAAP()

    f_DDE = DDE(filename)
    f_DDE.DDE()

    f_APAAC = APAAC(filename)
    f_APAAC.APAAC()

    f_ASDC = ASDC(filename)
    f_ASDC.ASDC()

    f_CTDC = CTDC(filename)
    f_CTDC.CTDC()

    f_CTDT = CTDT(filename)
    f_CTDT.CTDT()

    f_CTDD = CTDD(filename)
    f_CTDD.CTDD()

    f_QSO = QSO(filename)
    f_QSO.QSO()
########新特征##########
    f_DPC = DPC(filename)
    f_DPC.DPC()

    f_TPC = TPC(filename)
    f_TPC.TPC()

    f_SE = SE(filename)
    f_SE.SE()

    f_SER = SER(filename)
    f_SER.SER()

    f_SEP = SEP(filename)
    f_SEP.SEP()

    f_SOCN = SOCN(filename)
    f_SOCN.SOCN()

    ##########新特征结束############

    f_descriptors = calculate_molecular_descriptors(filename) # Initialize the class
    f_descriptors.MolecularDescriptors()
    descriptors_file = os.path.join(output_path, 'Molecular_Descriptors_features.csv')
    numpy.savetxt(descriptors_file, f_descriptors.encoding_array, fmt='%s', delimiter=',')

      # 获取分子描述符

    # 3. 将每个特征保存到单独的文件夹中
    # 保存 AAC 特征
    aac_file = os.path.join(output_path, 'AAC_features.csv')
    pd.DataFrame(f_AAC.encoding_array[1:], columns=f_AAC.encoding_array[0]).to_csv(aac_file, index=False)

    # 保存 CKSAAP 特征
    cksaap_file = os.path.join(output_path, 'CKSAAP_features.csv')
    pd.DataFrame(f_CKSAAP.encoding_array[1:], columns=f_CKSAAP.encoding_array[0]).to_csv(cksaap_file, index=False)

    # 保存 DDE 特征
    dde_file = os.path.join(output_path, 'DDE_features.csv')
    pd.DataFrame(f_DDE.encoding_array[1:], columns=f_DDE.encoding_array[0]).to_csv(dde_file, index=False)

    # 保存 APAAC 特征
    apaac_file = os.path.join(output_path, 'APAAC_features.csv')
    pd.DataFrame(f_APAAC.encoding_array[1:], columns=f_APAAC.encoding_array[0]).to_csv(apaac_file, index=False)

    # 保存 ASDC 特征
    asdc_file = os.path.join(output_path, 'ASDC_features.csv')
    pd.DataFrame(f_ASDC.encoding_array[1:], columns=f_ASDC.encoding_array[0]).to_csv(asdc_file, index=False)

    # 保存 CTDC 特征
    ctdc_file = os.path.join(output_path, 'CTDC_features.csv')
    pd.DataFrame(f_CTDC.encoding_array[1:], columns=f_CTDC.encoding_array[0]).to_csv(ctdc_file, index=False)

    # 保存 CTDT 特征
    ctdt_file = os.path.join(output_path, 'CTDT_features.csv')
    pd.DataFrame(f_CTDT.encoding_array[1:], columns=f_CTDT.encoding_array[0]).to_csv(ctdt_file, index=False)

    # 保存 CTDD 特征
    ctdd_file = os.path.join(output_path, 'CTDD_features.csv')
    pd.DataFrame(f_CTDD.encoding_array[1:], columns=f_CTDD.encoding_array[0]).to_csv(ctdd_file, index=False)

    # 保存 QSO 特征
    qso_file = os.path.join(output_path, 'QSO_features.csv')
    pd.DataFrame(f_QSO.encoding_array[1:], columns=f_QSO.encoding_array[0]).to_csv(qso_file, index=False)


    # 保存 DPC 特征
    qso_file = os.path.join(output_path, 'DPC_features.csv')
    pd.DataFrame(f_DPC.encoding_array[1:], columns=f_DPC.encoding_array[0]).to_csv(qso_file, index=False)

    # 保存 TPC 特征
    qso_file = os.path.join(output_path, 'TPC_features.csv')
    pd.DataFrame(f_TPC.encoding_array[1:], columns=f_TPC.encoding_array[0]).to_csv(qso_file, index=False)

    # 保存 SE 特征
    qso_file = os.path.join(output_path, 'SE_features.csv')
    pd.DataFrame(f_SE.encoding_array[1:], columns=f_SE.encoding_array[0]).to_csv(qso_file, index=False)

    # 保存 SER 特征
    qso_file = os.path.join(output_path, 'SER_features.csv')
    pd.DataFrame(f_SER.encoding_array[1:], columns=f_SER.encoding_array[0]).to_csv(qso_file, index=False)

    # 保存 SEP 特征
    qso_file = os.path.join(output_path, 'SEP_features.csv')
    pd.DataFrame(f_SEP.encoding_array[1:], columns=f_SEP.encoding_array[0]).to_csv(qso_file, index=False)

    # 保存 SOCN 特征
    qso_file = os.path.join(output_path, 'SOCN_features.csv')
    pd.DataFrame(f_SOCN.encoding_array[1:], columns=f_SOCN.encoding_array[0]).to_csv(qso_file, index=False)

    print(f"Features saved to {output_path}")

    print("\n开始整合特征文件...")

    # 获取输出目录下所有特征文件（按名称排序）
    all_files = sorted(glob.glob(os.path.join(output_path, "*_features.csv")))

    if not all_files:
        raise FileNotFoundError(f"No feature files found in {output_path}")

    # 读取第一个文件作为基准（包含Name和Label）
    combined_df = pd.read_csv(all_files[0])

    # 遍历剩余文件
    for file in all_files[1:]:
        # 读取当前文件并跳过前两列
        temp_df = pd.read_csv(file)
        feature_cols = temp_df.columns[2:]
        temp_data = temp_df.iloc[:, 2:].values

        # 处理列名冲突：添加文件名后缀保证唯一性
        temp_cols = []
        for col in feature_cols:
            if col in combined_df.columns:
                # 提取原始列名中的特征类型（如AAC_features.csv -> AAC）
                feature_name = os.path.basename(file).split('_features.csv')[0]
                new_col = f"{col}_{feature_name}"
            else:
                new_col = col
            temp_cols.append(new_col)

        # 创建临时DataFrame并重命名列
        temp_df = pd.DataFrame(temp_data, columns=temp_cols)

        # 合并数据（保留所有列）
        combined_df = pd.concat([combined_df, temp_df], axis=1)

        print(f"成功合并文件：{file} ({len(temp_cols)}个新特征)")

    # 保存最终结果
    final_file = os.path.join(output_path, 'Domain_features.csv')
    combined_df.to_csv(final_file, index=False)
    print(f"\n所有特征已整合到：{final_file}")


if __name__ == '__main__':
    # Define paths

    # train_path = r'E:\BBB\dif\BBppredict_data\train\train.fasta'
    # output_train_path = r'E:\BBB\dif\BBppredict_data\train'
    # #
    train_path = r'E:\yl\BBB\BBB\dif\BBppredict_data\324+324\val\val.fasta'
    output_train_path = r'E:\yl\BBB\BBB\test'

    # val_path = r'E:\BBB\new_test\train\train.fasta'
    # output_val_path = r'E:\BBB\new_test\train'
    #
    # val_path_1 = r'../data_new/neg_rename.fasta'
    # output_val_path_1 = r'../data_new/neg_features'

    # predict_path = r'E:\BBB\All_Data\new_generate\data4\Predict\predict.fasta.fasta'
    # output_predict_path =r'E:\BBB\All_Data\new_generate\data4\Predict'

    FeatureExtraction(train_path,output_train_path)
    # FeatureExtraction(val_path, output_val_path)
    # FeatureExtraction(val_path_1, output_val_path_1)
    # FeatureExtraction(predict_path, output_predict_path)



