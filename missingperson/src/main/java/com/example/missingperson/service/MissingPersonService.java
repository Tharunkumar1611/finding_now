package com.example.missingperson.service;

import com.example.missingperson.entity.MissingPersonEntity;
import com.example.missingperson.repository.MissingPersonRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class MissingPersonService {

    @Autowired
    private MissingPersonRepository missingPersonRepository;

    public List<MissingPersonEntity> getAllMissingPersons() {
        return missingPersonRepository.findAll();
    }

    public MissingPersonEntity saveMissingPerson(MissingPersonEntity missingPerson) {
        return missingPersonRepository.save(missingPerson);
    }

    public MissingPersonEntity updateMissingPerson(Long id, MissingPersonEntity updatedPerson) {
        Optional<MissingPersonEntity> existingPerson = missingPersonRepository.findById(id);
        if (existingPerson.isPresent()) {
            MissingPersonEntity person = existingPerson.get();
            person.setName(updatedPerson.getName());
            person.setAge(updatedPerson.getAge());
            person.setGender(updatedPerson.getGender());
            person.setLastSeenLocation(updatedPerson.getLastSeenLocation());
            person.setImage(updatedPerson.getImage());
            person.setReportedBy(updatedPerson.getReportedBy());
            return missingPersonRepository.save(person);
        } else {
            throw new RuntimeException("Missing person not found with id: " + id);
        }
    }

    public void deleteMissingPerson(Long id) {
        missingPersonRepository.deleteById(id);
    }
}
