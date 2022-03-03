#!/bin/bash

# NELL tasks
declare -a arr=(
athleteplaysinleague 
worksfor 
organizationhiredperson)
athleteplayssport 
teamplayssport 
personborninlocation 
athletehomestadium 
organizationheadquarteredincity 
athleteplaysforteam 
teamplaysinleague 
agentbelongstoorganization
personleadsorganization)

# Freebase tasks
#film@director@film
#film@film@language
#film@film@written_by
#location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division
#music@artist@origin
#organization@organization_founder@organizations_founded
#sports@sports_team@sport
#tv@tv_program@languages
#people@person@nationality
#

#Kinship tasks
#declare -a arr=(term18 term8 term3 term1 term19 term16 term15 term7 term17 term12 term11 term20 term25 term13 term4 term2 term0 term5 term9 term21 term22 term14 term24 term6 term1)

#declare -a arr=(people@person@place_of_birth)

#Countries tasks
#declare -a arr=(countries_S1 countries_S2 countries_S3)

for task_name in "${arr[@]}"
do
	echo $task_name
    	python story_generator.py $task_name nell
#    python story_generator.py $task_name fb

#    python story_generator.py $task_name nell data/NELL-995/entity_type/70/entity_type.txt			# partial entities
#    python story_generator.py $task_name nell data/NELL-995/entity_type/50/entity_type.txt			# partial entities
#    python story_generator.py $task_name nell data/NELL-995/entity_type/30/entity_type.txt			# partial entities

#    python story_generator.py $task_name fb data/FB15k-237/entity_type/70/entity_type.txt			# partial entities
#    python story_generator.py $task_name fb data/FB15k-237/entity_type/50/entity_type.txt			# partial entities
#    python story_generator.py $task_name fb data/FB15k-237/entity_type/30/entity_type.txt			# partial entities

done
