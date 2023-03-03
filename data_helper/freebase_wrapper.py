import json
from typing import Dict, List

from tqdm import tqdm

from common.common_utils import SENT_SPLIT

MIDSTART = ["m.", "g."]
ENG = " filter(!isliteral (?x) or lang (?x)='' or langmatches(lang(?x),'en')) "


def read_jsonl(path="test.jsonl", desc="", max_instances=None, total=None):
    with open(path, "r", encoding="utf-8") as f1:
        res = []
        _iter = tqdm(enumerate(f1), desc=desc, total=total) if desc else enumerate(f1)
        for idx, line in _iter:
            if max_instances and idx >= max_instances:
                break
            res.append(json.loads(line.strip()))
    return res


class FreebaseWrapper:
    """
    usage demo:
        fb = FreebaseWrapper(end_point="http://localhost:8890/sparql")
        es = Elasticsearch(hosts=["192.168.4.192:15005"], timeout=30, max_retries=10, retry_on_timeout=True)
        fb.set_es(es, index="freebase_desc")
    """

    def __init__(
        self,
        end_point="http://localhost:8890/sparql",
        prefix="PREFIX ns: <http://rdf.freebase.com/ns/> ",
        # es=Elasticsearch(hosts=["192.168.4.192:15005"], timeout=30, max_retries=10, retry_on_timeout=True)
        **kwargs,
    ):
        from SPARQLWrapper import JSON, SPARQLWrapper

        self.redis = kwargs.get("redis", None)
        self.es = None

        self.end_point = end_point
        self.prefix = prefix
        self.prefix_abbreviation = prefix.split(":")[0].split(" ")[-1] + ":"
        self.memory = None  # read all mids to memory
        self.sparql = SPARQLWrapper(end_point, returnFormat=JSON)
        self._current_source = ""

        # Meaningful description and type predicates
        self.predicates_description = [
            "common.topic.description",
            "base.schemastaging.nutrition_information.per_quantity_description",
            "common.webpage.description",
            "award.award_nomination.notes_description",
            "award.award_honor.notes_description",
            "base.schemastaging.food_energy_information.per_quantity_description",
            "biology.gene_ontology_group.description",
            "base.schemastaging.phone_sandbox.description",
            "base.schemastaging.listed_serving_size.quantity_description",
        ]
        self.predicates_description = [
            self.prefix_abbreviation + item for item in self.predicates_description
        ]

        # Entity description predicates
        self.predicates_type = [
            "type.object.name",
            "base.culturalevent.event.event_type",
            "base.biblioness.bibs_location.loc_type",
            "base.kwebbase.kwtopic.kwtype",
            "base.aareas.schema.administrative_area_type.stem_word",
            "base.artist_domain.ua_artwork.x_type",
            "base.lostbase.episode_character_relationship.relationship_type",
        ]
        self.predicates_type = [self.prefix_abbreviation + item for item in self.predicates_type]

        # CVT
        self.cvt_types = [
            "type.object.type",
        ]
        self.cvt_types = [self.prefix_abbreviation + item for item in self.cvt_types]
        self.es = None

    def set_es(
        self,
        host="localhost:9200",
        index="fb_desc_spacy_cased",
    ):
        from elasticsearch import Elasticsearch

        self.es = Elasticsearch(
            hosts=[host],
            timeout=30,
            max_retries=10,
            retry_on_timeout=True,
        )
        self.es_index = index

    def add_prefix_abbreviation(self, text):
        """
        Add prefix
        """
        text = str(text.strip())
        if text.startswith("www.freebase.com/m/"):
            text = self.prefix_abbreviation + text.replace("www.freebase.com/m/", "")
        elif not text.startswith(self.prefix_abbreviation):
            text = self.prefix_abbreviation + text
        return text

    def query(self, sparql_txt, return_bindings=True, _print=False):
        """
        Execute sparql
        """
        try:
            self.sparql.setQuery(sparql_txt)
            res_json = self.sparql.query().convert()
            if return_bindings:
                res_json = res_json["results"]["bindings"]
            return res_json
        except Exception as e:
            if _print:
                print(f"\nError: \n{sparql_txt}\n{str(e)}")
            return None

    def find_obj_by_id_and_pred(
        self,
        mid,
        limit=None,
        return_bindings=True,
        predicate="type.object.name",
        early_stop=None,
        return_list=False,
    ):
        """
        Given id, find corresponding text description and type.
        """
        if predicate == "description":
            predicates = self.predicates_description
        elif predicate == "type-name":
            predicates = self.predicates_type
        elif predicate == "type-type":
            predicates = self.cvt_types
        else:
            predicates = [self.add_prefix_abbreviation(predicate)]

        if early_stop and early_stop < 1:
            early_stop = None
        mid = self.add_prefix_abbreviation(mid)

        if return_list:
            types = []
        else:
            types = {}

        for index, pred in enumerate(predicates):
            if early_stop and index >= early_stop:
                break
            LIMIT = f"LIMIT {limit}" if limit else ""
            sparql_txt = f"{self.prefix}" + f"SELECT DISTINCT ?x WHERE {{ {mid} {pred} ?x . {ENG} }} {LIMIT}"
            res = self.query(sparql_txt=sparql_txt, return_bindings=return_bindings)
            if res:
                pred = pred.replace(self.prefix_abbreviation, "")
                if return_list:
                    types.extend([(pred, i["x"]["value"].strip('"')) for i in res])
                else:
                    types[pred] = [i["x"]["value"].strip('"') for i in res]
        return types

    def find_one_hop_by_id(self, _id, as_subj=True, as_obj=False, limit=10, desc=False, debug=False):
        """
        Given the head entity/tail entity, find the predicate, tail entity/head entity of one-hop.
        Note! Here is no optional!
        """
        _id = self.add_prefix_abbreviation(_id)
        LIMIT = f"limit {limit}" if limit else ""

        if as_subj == False and as_obj == False:
            raise ValueError("Cannot be all false")

        # out constant is head entity   in constant is tail entity
        res_list = []
        wheres = []
        # return description text or answer text
        if as_subj == True:
            STATEMENT = "SELECT DISTINCT ?p_out_1 ?obj_1 ?x"
            wheres.append(f"{_id} ?p_out_1 ?obj_1 .")
            wheres.append(f"?obj_1 {self.prefix_abbreviation}type.object.name ?x .")

            if desc:
                STATEMENT += " ?obj_1_desc"
                wheres.append(f"?obj_1 {self.prefix_abbreviation}common.topic.description ?obj_1_desc .")

            WHERE = "where { " + "\n".join(wheres) + " }"
            sparql_text = "\n".join([self.prefix, STATEMENT, WHERE, LIMIT])
            if debug:
                print(sparql_text)
            res_list += self.query(sparql_txt=sparql_text)

        wheres = []
        if as_obj == True:
            STATEMENT = "SELECT DISTINCT ?subj_1 ?subj_1_name ?p_in_1"
            wheres.append(f"?subj_1 ?p_in_1 {_id} .")
            wheres.append(f"?subj_1 {self.prefix_abbreviation}type.object.name ?subj_1_name .")

            if desc:
                STATEMENT += " ?obj_1_desc"
                wheres.append(f"?obj_1 {self.prefix_abbreviation}common.topic.description ?obj_1_desc .")

            WHERE = "where { " + "\n".join(wheres) + " }"
            sparql_text = "\n".join([self.prefix, STATEMENT, WHERE, LIMIT])
            if debug:
                print(sparql_text)
            res_list += self.query(sparql_txt=sparql_text)

        return res_list

    def cache_redis(self, path):
        with open(path, "r", encoding="utf-8") as f1:
            for line in tqdm(
                f1,
                total=47174184,
            ):
                line = json.loads(line.strip())
                self.redis.set(line[0], json.dumps(line[1:]))

    def load_mids(self, path, max_instances=None, total=None):
        """
        fb.load_mids("fb_mids.jsonl",max_instances=None,total=47174184)
        """
        if self.memory is None or (max_instances and max_instances > len(self.memory)):
            res = read_jsonl(path, desc="Loading mids", max_instances=max_instances, total=total)
            self.memory = {
                ll[0]: {
                    "name": ll[1].replace("@en", "").strip('"'),
                    "desc": ll[2].replace("@en", "").strip('"'),
                }
                for ll in res
            }

    def is_cvt(self, mid):
        res = self.is_topic_entity(mid)
        return not bool(res)

    def is_topic_entity(self, mid):
        res = self.get_mid_name(mid)
        return bool(res)

    def get_mid_name(self, mid):
        """
        add pathq special
        """
        if mid.startswith("None@"):
            return mid[5:].replace("_", " ")
        mid = mid.replace("<", "").replace(">", "").replace("http://rdf.freebase.com/ns/", "")
        res = ""

        # from memory
        if self.memory:
            if self._current_source != "memory":
                # print("current source: memory")
                self._current_source = "memory"

            res = self.memory.get(mid, {"name": "", "desc": ""})["name"].replace("@en", "").strip('"')
            return res

        # from ES
        if self.es:
            if self._current_source != "es":
                # print("current source: ES")
                self._current_source = "es"

            dsl = {"query": {"ids": {"values": [mid]}}}
            res = self.es.search(index=self.es_index, **dsl)["hits"]["hits"]
            if res:
                res = res[0]["_source"]["name"]
                return res

        # from redis
        if self.redis and not res:
            if self._current_source != "redis":
                print("current source: redis")
                self._current_source = "redis"

            res = self.redis.get(mid)
            if res:
                res = json.loads(res)
                return res[0].replace("@en", "").strip('"')
            else:
                return ""

        # from KG
        if self._current_source != "KG":
            # print("current source: KG")
            self._current_source = "KG"

        res = res if res else self.get_type_object_name(mid=mid)
        if len(res) > 0:
            res = res[0][1]
            return res
        else:
            return ""

    def get_mid_desc(self, mid):
        # from memory
        if self.memory:
            if self._current_source != "memory":
                print("current source: memory")
                self._current_source = "memory"

            res = self.memory.get(mid, {"name": "", "desc": ""})["name"].replace("@en", "").strip('"')
            return res

        # from ES
        if self.es:
            if self._current_source != "es":
                print("current source: ES")
                self._current_source = "es"

            dsl = {"query": {"ids": {"values": [mid]}}}
            res = self.es.search(index=self.es_index, **dsl)["hits"]["hits"]
            if res:
                res = res[0]["_source"]["description"]
                return res
            else:
                return ""

        # from redis
        if self.redis:
            if self._current_source != "redis":
                print("current source: redis")
                self._current_source = "redis"

            res = self.redis.get(mid)
            if res:
                res = json.loads(res)
                return res[1].replace("@en", "").strip('"')
            else:
                return ""

        # from KG
        if self._current_source != "KG":
            print("current source: KG")
            self._current_source = "KG"
        res = self.get_desc(mid=mid)
        if len(res) > 0:
            res = res[0][1]
            return res
        else:
            return ""

    def is_topic(self, mid):
        return True if self.get_mid_name(mid) else False

    def get_type_object_name(self, mid) -> List[Dict]:
        res = self.find_obj_by_id_and_pred(
            mid=mid,
            limit=None,
            return_bindings=True,
            predicate="type.object.name",
            early_stop=None,
            return_list=True,
        )
        return res

    def get_type_object_type(self, mid):
        res = self.find_obj_by_id_and_pred(
            mid=mid,
            limit=None,
            return_bindings=True,
            predicate="type.object.type",
            early_stop=None,
            return_list=True,
        )
        return res

    def get_desc(self, mid) -> List[Dict]:
        """
        must common.topic.description
        """
        res = self.find_obj_by_id_and_pred(
            mid=mid,
            limit=None,
            return_bindings=True,
            predicate="description",
            early_stop=1,
            return_list=True,
        )
        return res[0][1] if res else ""

    def get_desc_spacy(self, mid):
        if mid[:2] not in MIDSTART:
            return ""
        try:
            dsl = {"query": {"ids": {"values": [mid]}}}
            res = self.es.search(index=self.es_index, **dsl)["hits"]["hits"]
            if res:
                desc_first = res[0]["_source"]["desc"].split(SENT_SPLIT)[0].split("\\n")[0]
                return desc_first
        except Exception as e:
            print(e)
        return ""

    def get_mid_type_matched(self, mid):
        """
        Reference previous work, count the number of freebase types in the first paragraph of Wikipedia, and the most one is type.
        """
        if mid[:2] not in MIDSTART:
            return ""
        try:
            dsl = {"query": {"ids": {"values": [mid]}}}
            res = self.es.search(index="type_match", **dsl)["hits"]["hits"]
            if res:
                return res[0]["_source"]["type_match"]
        except Exception as e:
            print(e)
        return ""

    def get_2_hop_out(self, mid):
        pass

    def get_3_hop_out(self, mid):
        pass

    def entity_search(self, ent_name):
        """
        Update, exact mode. At least start with matching.
        """
        # ent_name = ent_name.lower()
        dsl = {"query": {"bool": {"must": [{"match_phrase": {"name": ent_name}}]}}}
        res = self.es.search(index=self.es_index, **dsl)["hits"]["hits"]
        res = [i for i in res if i["_source"]["name"].lower().startswith(ent_name.lower())]
        return res
